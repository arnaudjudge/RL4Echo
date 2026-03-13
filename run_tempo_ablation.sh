#python rl4echo/runner.py train=False +test_from_ckpt=/home/local/USHERBROOKE/juda2901/dev/RL4Echo/checkpoints/hp_ablation/sigma1.ckpt +model.temp_files_path=sigma1 logger_run_name=sigma1_test
#python rl4echo/runner.py train=False +test_from_ckpt=/home/local/USHERBROOKE/juda2901/dev/RL4Echo/checkpoints/hp_ablation/sigma10.ckpt +model.temp_files_path=sigma10 logger_run_name=sigma10_test
#python rl4echo/runner.py train=False +test_from_ckpt=/home/local/USHERBROOKE/juda2901/dev/RL4Echo/checkpoints/hp_ablation/sigma50.ckpt +model.temp_files_path=sigma50 logger_run_name=sigma50_test
#python rl4echo/runner.py train=False +test_from_ckpt=/home/local/USHERBROOKE/juda2901/dev/RL4Echo/checkpoints/hp_ablation/sigma100-2.ckpt +model.temp_files_path=sigma100-2 logger_run_name=sigma100-2_test
python rl4echo/runner.py train=False +test_from_ckpt=/home/local/USHERBROOKE/juda2901/dev/RL4Echo/checkpoints/hp_ablation/sigma500.ckpt +model.temp_files_path=sigma500 logger_run_name=sigma500_test
