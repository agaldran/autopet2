# srun --nodes=1 --gres=gpu:1 --exclude=node020,node021,node022,node023 --ntasks=1 --cpus-per-task=16 --partition=high --pty bash -i

##########################################################################
python train_class.py --projection x --model swin --cycle_lens 5/3 --pretrained True  --save_path mipx_swin_wd3_f1 --fold 1 --weight_decay 1e-3
python train_class.py --projection y --model swin --cycle_lens 5/3 --pretrained True  --save_path mipy_swin_wd3_f1 --fold 1 --weight_decay 1e-3

python train_class.py --projection x --model swin --cycle_lens 5/3 --pretrained True  --save_path mipx_swin_wd3_f2 --fold 2 --weight_decay 1e-3
python train_class.py --projection y --model swin --cycle_lens 5/3 --pretrained True  --save_path mipy_swin_wd3_f2 --fold 2 --weight_decay 1e-3

python train_class.py --projection x --model swin --cycle_lens 5/3 --pretrained True  --save_path mipx_swin_wd3_f3 --fold 3 --weight_decay 1e-3
python train_class.py --projection y --model swin --cycle_lens 5/3 --pretrained True  --save_path mipy_swin_wd3_f3 --fold 3 --weight_decay 1e-3

python train_class.py --projection x --model swin --cycle_lens 5/3 --pretrained True  --save_path mipx_swin_wd3_f4 --fold 4 --weight_decay 1e-3
python train_class.py --projection y --model swin --cycle_lens 5/3 --pretrained True  --save_path mipy_swin_wd3_f4 --fold 4 --weight_decay 1e-3

python train_class.py --projection x --model swin --cycle_lens 5/3 --pretrained True  --save_path mipx_swin_wd3_f5 --fold 5 --weight_decay 1e-3
python train_class.py --projection y --model swin --cycle_lens 5/3 --pretrained True  --save_path mipy_swin_wd3_f5 --fold 5 --weight_decay 1e-3
