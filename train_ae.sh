printf "\nRUNNING python main.py --ae --train --phase 0 --iteration 8000000 --sample_dir sample_res_table/all_vox256_img0_16 --sample_vox_size 16\n"
python main.py --ae --train --phase 0 --iteration 8000000 --sample_dir sample_res_table/all_vox256_img0_16 --sample_vox_size 16
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_16 --start 0 --end 16\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_16 --start 0 --end 16
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_16 --start 2988 --end 3004\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_16 --start 2988 --end 3004

printf "\nRUNNING python main.py --ae --train --phase 0 --iteration 8000000 --sample_dir sample_res_table/all_vox256_img0_32 --sample_vox_size 32\n"
python main.py --ae --train --phase 0 --iteration 8000000 --sample_dir sample_res_table/all_vox256_img0_32 --sample_vox_size 32
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_32 --start 0 --end 16\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_32 --start 0 --end 16
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_32 --start 2988 --end 3004\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_32 --start 2988 --end 3004

printf "\nRUNNING python main.py --ae --train --phase 0 --iteration 16000000 --sample_dir sample_res_table/all_vox256_img0_64 --sample_vox_size 64\n"
python main.py --ae --train --phase 0 --iteration 16000000 --sample_dir sample_res_table/all_vox256_img0_64 --sample_vox_size 64
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_64 --start 0 --end 16\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_64 --start 0 --end 16
printf "\nRUNNING python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_64 --start 2988 --end 3004\n"
python main.py --ae --phase 0 --sample_dir sample_res_table/all_vox256_img0_64 --start 2988 --end 3004

printf "\nRUNNING python main.py --ae --train --phase 1 --iteration 16000000 --sample_dir sample_res_table/all_vox256_img1 --sample_vox_size 64\n"
python main.py --ae --train --phase 1 --iteration 16000000 --sample_dir sample_res_table/all_vox256_img1 --sample_vox_size 64
printf "\nRUNNING python main.py --ae --phase 1 --sample_dir sample_res_table/all_vox256_img1 --start 0 --end 16\n"
python main.py --ae --phase 1 --sample_dir sample_res_table/all_vox256_img1 --start 0 --end 16
printf "\nRUNNING python main.py --ae --phase 1 --sample_dir sample_res_table/all_vox256_img1 --start 2988 --end 3004\n"
python main.py --ae --phase 1 --sample_dir sample_res_table/all_vox256_img1 --start 2988 --end 3004

printf "\nRUNNING python main.py --ae --getz\n"
python main.py --ae --getz
