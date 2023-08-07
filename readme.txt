test
python main.py --save_dir ./eval/compression_canshu/easy_mode05422   --reset True  --log_file_name eval.log  --eval True  --eval_save_results True  --usegmm True  --num_workers 0  --dataset SATELITE  --dataset_dir ../compression-TTSR/satelite  --model_path ./pretrained/model_x2_compress.pt

eval TTSR+JPEG
python main.py --save_dir ./eval/Complexity/adaptive --reset True  --log_file_name eval.log  --eval True  --eval_save_results True  --usegmm True  --num_workers 0  --dataset SATELITE  --dataset_dir ../compression-TTSR/satelite  --model_path ./pretrained/WithGMM_iter274.pt

eval TTSR+GMM
python main.py --save_dir ./eval/TTSR_GMM --reset True  --log_file_name eval.log  --eval True  --eval_save_results True  --usegmm True  --num_workers 0  --dataset SATELITE  --dataset_dir ../compression-TTSR/satelite  --model_path ./pretrained/model_x2_compress.pt


train
python main.py --save_dir ./train/SATELLITE/TTSR+GMM  --reset True  --log_file_name train.log  --num_gpu 1  --num_workers 0  --dataset SATELITE  --dataset_dir G:/WHW/RS_compress/satellite  --n_feats 64  --lr_rate 1e-4  --lr_rate_dis 1e-4  --lr_rate_lte 1e-5  --rec_w 1  --per_w 1e-2  --tpl_w 1e-2  --adv_w 1e-3  --batch_size 2  --num_init_epochs 10  --num_epochs 200  --print_every 600  --save_every 10  --train_crop_size 32




