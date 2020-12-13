printf "\npython main.py --svr --train --epoch 1000 --sample_dir samples/all_vox256_img2\n"
python main.py --svr --train --epoch 1000 --sample_dir samples/all_vox256_img2
printf "\npython main.py --svr --sample_dir samples/all_vox256_img2 --start 0 --end 16\n"
python main.py --svr --sample_dir samples/all_vox256_img2 --start 0 --end 16
printf "\npython main.py --svr --sample_dir samples/all_vox256_img2 --start 2988 --end 3004\n"
python main.py --svr --sample_dir samples/all_vox256_img2 --start 2988 --end 3004