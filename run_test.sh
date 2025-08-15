
## Run testing of the trained models to produce the calibrated RGB segments
# The --num_test values correspond to the number of segments found from that PJ in our dataset

# Model C25_PJ15
python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C25_PJ15 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C25_PJ15_TEST_PJ13 --mode sb --eval --phase test --num_test 1334 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 13 --cycles 25

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C25_PJ15 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C25_PJ15_TEST_PJ14 --mode sb --eval --phase test --num_test 1139 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 14 --cycles 25

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C25_PJ15 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C25_PJ15_TEST_PJ15 --mode sb --eval --phase test --num_test 1693 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 15 --cycles 25

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C25_PJ15 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C25_PJ15_TEST_PJ16 --mode sb --eval --phase test --num_test 809 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 16 --cycles 25


# Model C26_PJ18
python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ18 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ18_TEST_PJ17 --mode sb --eval --phase test --num_test 1326 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 17 --cycles 26

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ18 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ18_TEST_PJ18 --mode sb --eval --phase test --num_test 1574 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 18 --cycles 26

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ18 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ18_TEST_PJ19 --mode sb --eval --phase test --num_test 991 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 19 --cycles 26


# Model C26_PJ20
python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ20_TEST_PJ20 --mode sb --eval --phase test --num_test 1855 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ20_TEST_PJ21 --mode sb --eval --phase test --num_test 2306 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 21 --cycles 26

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ20_TEST_PJ22 --mode sb --eval --phase test --num_test 2134 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 22 --cycles 26

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ20_TEST_PJ23 --mode sb --eval --phase test --num_test 1573 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 23 --cycles 26

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C26_PJ20 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C26_PJ20_TEST_PJ24 --mode sb --eval --phase test --num_test 1913 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 24 --cycles 26


# Model C27_PJ27
python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ25 --mode sb --eval --phase test --num_test 1869 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 25 --cycles 27

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ26 --mode sb --eval --phase test --num_test 1212 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 26 --cycles 27

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ27 --mode sb --eval --phase test --num_test 1528 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 27 --cycles 27

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ28 --mode sb --eval --phase test --num_test 1935 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 28 --cycles 27

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ29 --mode sb --eval --phase test --num_test 1346 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 29 --cycles 27

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C27_PJ27 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C27_PJ27_TEST_PJ30 --mode sb --eval --phase test --num_test 1045 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 40 --use_zone_pairs --fix_time_bug --PJs 30 --cycles 27


# Model C28_PJ33
python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ31 --mode sb --eval --phase test --num_test 924 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 31 --cycles 28

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ32 --mode sb --eval --phase test --num_test 2085 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 32 --cycles 28

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ33 --mode sb --eval --phase test --num_test 1734 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 33 --cycles 28

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ34 --mode sb --eval --phase test --num_test 1691 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 34 --cycles 28

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ35 --mode sb --eval --phase test --num_test 1670 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 35 --cycles 28

python test.py --dataroot ./datasets/junocam_calibration_all_npy --name junocam_calibration_C28_PJ33 --checkpoints_dir ./checkpoints --out_name junocam_calibration_C28_PJ33_TEST_PJ36 --mode sb --eval --phase test --num_test 1700 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3 --epoch 50 --use_zone_pairs --fix_time_bug --PJs 36 --cycles 28



## Run testing of the UV,M prediction trained models
# We assume that the calibrated RGB images were placed under datasets/calibrated_all_npy.
# The sequence of cmds is very similar to the RGB case except for: 1) the dataroot needs to point to the calibrated npy files, and 2) input_nc and output_nc have a value of 2.
#We provide one example:
python test.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_C25_PJ15 --checkpoints_dir ./checkpoints --out_name junocam_calibration_UVM_C25_PJ15_TEST_PJ13 --mode sb --eval --phase test --num_test 1334 --dataset_mode unaligned_npy --input_nc 2 --output_nc 2 --epoch latest --use_zone_pairs --fix_time_bug --PJs 13 --cycles 25


# Run test on the GRS-specific model
python test.py --dataroot ./datasets/calibrated_all_npy --name junocam_calibration_UVM_GRS_images --checkpoints_dir ./checkpoints --mode sb --eval --phase test --num_test 242 --dataset_mode unaligned_npy --input_nc 2 --output_nc 2 --epoch latest --fix_time_bug --zones GRS_images















