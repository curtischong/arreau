.PHONY: train

train:
	python main_diffusion.py --num_timesteps=300 --gpus=1 --radius=5 --num_workers=-1 --experiment_name=full_run

generate:
	python main_diffusion_generate.py --model_path="/Users/curtischong/Documents/dev/arreau/logs/PONITA-alexandria/axog3rde/checkpoints/epoch=2759-step=2760.ckpt"
