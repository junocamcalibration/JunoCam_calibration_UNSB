import os.path
from data.base_dataset import BaseDataset, get_transform, normalize_npy
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import torch
import math


class AlignedNPYDataset(BaseDataset):
    '''
    This class is to use the npy files of the same domain (HST) to perform RGB to UV,M mapping
    '''

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # HST is domain B in our settings
        path = os.path.join(opt.dataroot, opt.phase + 'B')

        # During testing if test folders are not there, then switch to the train folders
        # Assume train folders are always there
        if os.path.exists(path):
            self.dir = path
        else:
            self.dir = os.path.join(opt.dataroot, 'trainB')

        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))   # load images from '/path/to/data/trainB'

        ## Filter A and B based on given zones
        if len(self.opt.zones)>0: # if no zones are given then don't filter
            self.paths = self.filter_data_by_zones(self.paths, domain="hst")

        ## Filter B based on given Cycles
        if len(self.opt.cycles)>0:
            self.paths = self.filter_data_by_map_id(self.paths, map_id_list=self.opt.cycles)

        self.size = len(self.paths)  # get the size of dataset B


    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size


    def filter_data_by_map_id(self, paths, map_id_list):
        filtered_paths = []
        for i in range(len(paths)):
            map_id = paths[i].split('/')[-1].split('_')[2]
            if map_id in map_id_list:
                filtered_paths.append(paths[i]) 
        return filtered_paths
                

    def filter_data_by_zones(self, paths, domain):
        filtered_paths = []
        if "GRS_images" in self.opt.zones:
            # If GRS_images is in the zones option then we use the predefined list of ids
            with open(self.opt.dataroot+"/"+domain+"_GRS_images.txt", "r") as f:
                img_list = f.readlines()
            img_list = [x.split('\n')[0] for x in img_list]
            for i in range(len(paths)):
                data_img_id = paths[i].split('/')[-1].split('.')[0]
                if data_img_id in img_list:
                    filtered_paths.append(paths[i])

        else:
            for i in range(len(paths)):
                zone_path = paths[i].split('/')[-1].split('_')[0]
                if zone_path in self.opt.zones:
                    filtered_paths.append(paths[i]) 
        return filtered_paths


    def __getitem__(self, index):
        '''
        Return an aligned pair where the source domain is R,B and the target is UV,Methane from HST npy segments
        '''
        index_B = index % self.size
        B_path = self.paths[index_B]  # make sure index is within then range

        # npy data are already normalized between 0...1        
        B_img = np.load(B_path)
        B_img = torch.tensor(B_img) # already in: n_channels x H x W

        ## Do correction for the Methane band
        # GRS is supposed to have the highest values in Methane band and its values are between 0.12-0.18
        # So probably a 5.0 multiplier is fine
        B_img[4,:,:] = B_img[4,:,:] * 5.0

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        # convert=False because inputs are already tensors and we use a later function to normalize
        transform = get_transform(modified_opt, convert=False)
        B = transform(B_img)

        # Custom function to normalize inputs of arbitrary number of channels to [-1, 1]
        B = normalize_npy(B) # 5 x H x W -- UV, B, G, R, Methane 

        # Re-Order channels of B to: (B, G, R, UV, Methane)
        order = torch.tensor([1, 2, 3, 0, 4])
        B = B[order, :, :]

        # Use B,R channels as source and UV,Methane as targets
        source = B[torch.tensor([0,2]),:,:].clone()
        target = B[3:,:,:].clone()

        item = {'A': source, 'B': target, 'B_orig': B, 'A_paths': B_path, 'B_paths': B_path}

        return item