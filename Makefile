.PHONY: train generate prep-datasets

prep-datasets:
	python diffusion/prep_datasets.py

train:
	python main_diffusion.py --num_timesteps=1000 --gpus=1 --radius=5 --num_workers=-1 --max_neighbors=8 --batch_size=350 --lr=0.0003

generate:
	python main_diffusion_generate.py --model_path="models/last.ckpt"

profile:
	python main_diffusion.py --num_timesteps=100 --gpus=0 --radius=5 --num_workers=-1 --is_local_dev=true --max_neighbors=8 --epochs=1000 --profiler=advanced
