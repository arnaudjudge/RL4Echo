#python rl4echo/runner.py seed=123 model.ckpt_path=Ensemble_seed123.ckpt datamodule=cardinal_3d
python rl4echo/runner.py seed=123 +model.load_from_ckpt=Ensemble_seed123.ckpt +model.do_unc=True datamodule=icardio_3d train=false model.tta=False
#python rl4echo/runner.py seed=456 model.ckpt_path=Ensemble_seed456.ckpt datamodule=cardinal_3d
python rl4echo/runner.py seed=456 +model.load_from_ckpt=Ensemble_seed456.ckpt +model.do_unc=True datamodule=icardio_3d train=False model.tta=False
#python rl4echo/runner.py seed=789 model.ckpt_path=Ensemble_seed789.ckpt datamodule=cardinal_3d
python rl4echo/runner.py seed=789 +model.load_from_ckpt=Ensemble_seed789.ckpt +model.do_unc=True datamodule=icardio_3d train=False model.tta=False

