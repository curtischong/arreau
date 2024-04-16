# Arreau

This is a diffusion model to generate new crystal materials!

This repo was heavily inspired by:
- [Mattergen](https://arxiv.org/pdf/2312.03687.pdf) for the overall idea and the lattice loss.
- [Mofdiff](https://github.com/microsoft/MOFDiff) for the atomic diffusion and fractional coordinate loss.
- [Mace](https://github.com/ACEsuit/mace) for helper functions (e.g. calculate periodic boundaries), and teaching me about equivariant architectures.
- In addition, the main model architecture is a fork of the [Ponita](https://github.com/ebekkers/ponita/) model.

### Computational statistics:
- The overal model is 1.1 million parameters. It is 18 Mb in size.
- The model is trained on a single A10 GPU with 24GB of VRAM. The score matching validation loss converges to around ~0.03 in around 30 minutes.
- We used the PBE functional of the [Alexandria dataset](https://archive.materialscloud.org/record/2023.71) (A new dataset of 415k stable and metastable materials calculated with the PBEsol and SCAN functionals).
- Training was performed using only 300k samples (72% of the dataset), and the other 115k were used for the validation/testing set

### Installation

1. Create a python environment to store the dependencies:
`python3.11 -m venv venv`

2. Activate the environment:
`source venv/bin/activate`

3. Install the dependencies:
- Note: We have different installations for different operating systems and for machines with/without GPUs.
- The reason why is because torch_cluster and torch_scatter want you to download their pre-compiled wheels for your system. However, these packages do not have the pre-compiled wheels available for some systems like MacOS. (So the cpu_requirements.txt points to a different repo that your computer can use to build and download the wheels).

    - If you have a GPU, run this command:
        ```
        pip install -r requirements.txt
        ```

    - If you want to install this on your local computer (especially for mac computers), run this command:
        ```
        pip install -r cpu_requirements.txt
        ```

4. Please install the Ruff linter VSCode extension
- So your changes are linted properly
- Please set the Default Formatter VSCode setting to use Ruff

4.5 setup githooks so we will run ruff before committing
- `git config --local core.hooksPath .githooks/`

### Weights:

You can download pre-trained weights [here](https://drive.google.com/drive/folders/1y84gdxGfzeN-RU8DyVpPvCXop7Xf5slu)


### Development tips:
Learn how diffusion works by watching [this video](https://www.youtube.com/watch?v=wMmqCMwuM2Q)

`main_diffusion.py` is the main file that trains the model.
- `diffusion.py` is the model definition.
- `diffusion_loss.py` is where the loss function is defined. This loss was heavily inspired by the Mofdiff paper.
- When you train the model, it generates pytorch lightning checkpoints. e.g. `checkpoints/last.ckpt`. These are all the necessary weights (and additional variables) each model needs to perform inference (or further training!).

`main_diffusion_generate.py` is the main file that generates new crystals.

You can run the Makefile to call these files.
e.g: `make train` or `make generate`

### Optional:

If you want to try relaxing the generated crystals, you can use Mace to relax the final configurations.
Note: Since the diffusion was trained on the relaxed crystals, you typically don't need to relax the generated crystals.
You can download the mace-mp-0 model

Download mace-mp-0 [here](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model)
- View all releases [here](https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0)

install the ipykernel so you can import python functions (in this repo) in jupyter notebooks: python -m ipykernel install --user --name=venv
- It works because this repo is saved as a dependency in venv