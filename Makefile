.PHONY: train

train:
	python main_diffusion.py --num_timesteps=300 --gpus=1 --radius=5 --num_workers=-1

generate:
	python main_diffusion_generate.py
