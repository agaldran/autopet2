
# srun --nodes=1 --gres=gpu:1 --exclude=node020,node021,node022,node023 --ntasks=1 --cpus-per-task=16 --partition=high --pty bash -i

python train_seg_pos.py --path_csv_train data/ --save_path swin_tiny_ultralong_notest_F1 --cycle_lens 50/5 --fold 1 --notest True
python train_seg_pos.py --path_csv_train data/ --save_path swin_tiny_ultralong_notest_F2 --cycle_lens 50/5 --fold 2 --notest True
python train_seg_pos.py --path_csv_train data/ --save_path swin_tiny_ultralong_notest_F3 --cycle_lens 50/5 --fold 3 --notest True
python train_seg_pos.py --path_csv_train data/ --save_path swin_tiny_ultralong_notest_F4 --cycle_lens 50/5 --fold 4 --notest True
python train_seg_pos.py --path_csv_train data/ --save_path swin_tiny_ultralong_notest_F5 --cycle_lens 50/5 --fold 5 --notest True


