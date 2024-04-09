.PHONY: train generate prep-datasets

prep-datasets:
	python diffusion/prep_datasets.py

train:
	python main_diffusion.py --num_timesteps=300 --gpus=1 --radius=5 --num_workers=-1 --experiment_name=no-lattice --max_neighbors=24

generate:
	python main_diffusion_generate.py --model_path="models/last.ckpt"
