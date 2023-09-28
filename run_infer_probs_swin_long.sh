
# srun --nodes=1 --gres=gpu:1 --exclude=node020,node021,node022,node023 --ntasks=1 --cpus-per-task=16 --partition=high --pty bash -i
# srun --nodes=1 --gres=gpu:1 --exclude=node031 --ntasks=1 --cpus-per-task=24 --partition=high --pty bash -i
#srun --nodes=1 --gres=gpu:1 --exclude=node031 --ntasks=1 --cpus-per-task=24 --partition=medium --pty bash -i

python inf_probs.py --experiment_path experiments/swin_tiny_long_F1/
python inf_probs.py --experiment_path experiments/swin_tiny_long_F2/
python inf_probs.py --experiment_path experiments/swin_tiny_long_F3/
python inf_probs.py --experiment_path experiments/swin_tiny_long_F4/
python inf_probs.py --experiment_path experiments/swin_tiny_long_F5/

python inf_probs.py --experiment_path experiments/swin_tiny_long_F1/ --test True
python inf_probs.py --experiment_path experiments/swin_tiny_long_F2/ --test True
python inf_probs.py --experiment_path experiments/swin_tiny_long_F3/ --test True
python inf_probs.py --experiment_path experiments/swin_tiny_long_F4/ --test True
python inf_probs.py --experiment_path experiments/swin_tiny_long_F5/ --test True
