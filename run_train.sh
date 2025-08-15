
## Training models for calibrating JunoCam RGB

# PJ15 -- PJs (13 15 16 17) -- Cycle 25
python train.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C25_PJ15 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3 --use_zone_pairs --fix_time_bug --PJs 13 15 16 17 --cycles 25

# PJ18 -- PJs (16 17 18 19 20) -- Cycle 26
python train.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ18 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3 --use_zone_pairs --fix_time_bug --PJs 16 17 18 19 20 --cycles 26

# PJ20 -- PJs (18, 19, 20, 21, 22) -- Cycle 26
python train.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3 --use_zone_pairs --fix_time_bug --PJs 18 19 20 21 22 --cycles 26

# PJ27 -- PJs (25 26 27 28 29) -- Cycle 27
python train.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3 --use_zone_pairs --fix_time_bug --PJs 25 26 27 28 29 --cycles 27

# PJ33 -- PJs (31 32 33 34 35) -- Cycle 28
python train.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3 --use_zone_pairs --fix_time_bug --PJs 31 32 33 34 35 --cycles 28



## Training models for predicting UV and Methane channels from the calibrated images
# We assume that the calibrated RGB images were placed under datasets/calibrated_all_npy.

# UVM PJ15 -- PJs (13 15 16 17) -- Cycle 25
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C25_PJ15 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --use_zone_pairs --fix_time_bug --PJs 13 15 16 17 --cycles 25

# UVM PJ18 -- PJs (16 17 18 19 20) -- Cycle 26
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C26_PJ18 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --use_zone_pairs --fix_time_bug --PJs 16 17 18 19 20 --cycles 26

# UVM PJ20 -- PJs (18, 19, 20, 21, 22) -- Cycle 26
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C26_PJ20 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --use_zone_pairs --fix_time_bug --PJs 18 19 20 21 22 --cycles 26

# UVM PJ27 -- PJs (25 26 27 28 29) -- Cycle 27
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C27_PJ27 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --use_zone_pairs --fix_time_bug --PJs 25 26 27 28 29 --cycles 27

# UVM PJ33 -- PJs (31 32 33 34 35) -- Cycle 28
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C28_PJ33 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --use_zone_pairs --fix_time_bug --PJs 31 32 33 34 35 --cycles 28

# GRS-specifc model for predicting UV and Methane
python train.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_GRS_images --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2 --fix_time_bug --zones GRS_images







