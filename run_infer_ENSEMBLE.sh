
# srun --nodes=1 --gres=gpu:1 --exclude=node020,node021,node022,node023 --ntasks=1 --cpus-per-task=16 --partition=high --pty bash -i
python inf_class_test_ensemble.py
python inf_probs_test_ensemble.py --experiment_path experiments/swin_tiny_long_F







