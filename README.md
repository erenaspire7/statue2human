# Automatic Colouring of Marble Statues

## Overview

This repository contains all relevant code used in the training and evaluation of the colourizer model. Due to compute limits, instructions for the CUT and CycleGAN are stored in [Colab](#training) but can be replicated locally.

## Setup

### Prerequisites

The python version used for this project is 3.12. Install python-3.12 and create a virtual environment with the dependencies specified in the requirements.txt file.

#### Install Python

- Linux - `sudo apt install python-3.12 python3.12-venv python3.12-dev`
- Windows / Mac - (https://www.python.org/downloads/release/python-3127/)

Note: Additional PATH setup will need to be done for Windows and Mac. Please view

- Mac (https://www.xda-developers.com/how-add-python-path-macos/)
- Windows (https://realpython.com/add-python-to-path/)

#### Project Setup

##### Create a new environment

```
python3.12 -m venv venv
```

##### Activate the environment

```
# Mac OS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

N.B: Windows might disable scripts by default, which will prevent the virtual environment from being activated. To rectify, run this command

```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```

##### Install the Libraries

```
pip install -r requirements.txt
```

## Training

Training the CycleGAN and the CUT models requires a large amount of compute. This was done in Google Colab. Note that batch size used was 8, however if there is a memory error, this value should be reduced. Training will take more time if the batch size is reduced.

Futher instructions are available at

- CycleGAN (https://colab.research.google.com/drive/1B0BOel2-7J21biXe6FTz6W1SOhL23SkF?usp=sharing)

- CUT (https://colab.research.google.com/drive/15oEKboltISfjUXMDGvCjqU-64tPsf-lR?usp=sharing)

Note that the commands present in the colab notebooks can also be adapted to any system. Windows users might need to follow https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-installation.

#### Training the colourizer network

The paired dataset is located at [link](https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/colourization-dataset-v2.zip). When extracted, there will be a `human` and a `statue-v2` folder. These needs to be referenced in an .env file. Use the `.env.example` file as a reference for your `.env` file and write as follows.

Make sure to change to path to reflect the absolute path location of the data.

```
# path to the statue folder
STATUE_PATH="/home/erenaspire7/repos/honours-project/keras-colorizer/data/statue-v2"

# path to the human folder
HUMAN_PATH="/home/erenaspire7/repos/honours-project/keras-colorizer/data/human"

BATCH_SIZE=50
```

Then you can run the colourizer training script with

```
python colourizer/main.py
```

N.B: The compute requirements might be too high during traing. Either reduce the `BATCH_SIZE` variable or Run on [Colab](https://colab.research.google.com/drive/1jvUo6M7xOgS-WApA_ohdYtIjC5JgnM31?usp=sharing) instead.

## Evaluation

The project contains three pre-trained models located at `colourizer/models`.

If cloned from github, download the models from

- https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/AutoEncoder+Models/colorizer-clean.keras
- https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/AutoEncoder+Models/colorizer-noisy.keras
- https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/AutoEncoder+Models/colorizer-v3.keras

In order to colourize your own statues, add these two variables to your `.env` file.

```
# path to where the results should be saved
RESULTS_PATH="/home/erenaspire7/repos/honours-project/compiled-project/results"

# the colourizer model of your choice, can be any of the pre-trained models or your own trained model
COLOURIZER_MODEL="/home/erenaspire7/repos/honours-project/compiled-project/models/colorizer-clean.keras"
```

Then you can the run colourization script with

```
python colourizer/evaluate.py --path <path-to-image>
```

This will output a colourized statue in your previously defined `RESULTS` folder.

#### CycleGAN evaluation

Download / Clone the repo from https://github.com/erenaspire7/pytorch-CycleGAN-and-pix2pix and setup a virtual environement as shown in [link](#setup).

Then download the pre-trained model at https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/CycleGAN/checkpoints.zip and unzip into the downloaded repo.

Within the repo, run this command for a single image

```
!python scripts/transform.py --name statue2human \
	--single_image --image_path /content/image.jpg --AtoB true \
	--results_path <folder-path>
```

and the command below for transforming multiple images.

N.B: (A) refers to a statue, and (B) refers to a human, hence setting AtoB as false performs a human2statue transformation.

```
!python scripts/transform.py --name statue2human \
	--multiple_images --image_dir /home/erenaspire7/repos/honours-project/pytorch-CycleGAN-and-pix2pix/statue-data --AtoB true \
	--results_path <folder-path>
```

#### CUT evaluation

Download the repo from https://github.com/erenaspire7/contrastive-unpaired-translation and setup a virtual environement as shown in [link](#setup).

Download the pre-trained model at https://erenaspire7-gan-dataset.s3.eu-west-1.amazonaws.com/CUT-model/pretrained_checkpoints.zip and unzip into the downloaded repo.

Within the repo, run this command for a single image

```
!python transform.py --name statue2human_CUT \
	--load_size 256 \
	--single_image --image_path /content/image.jpg \
	--results_path <folder-path>
```

and the command below for transforming multiple images.

```
!python transform.py --name statue2human_CUT \
	--load_size 256 \
	--multiple_images --image_dir /content/images \
	--results_path <folder-path>
```

## References / Sources

- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://github.com/taesungp/contrastive-unpaired-translation
- https://github.com/LS4GAN/uvcgan2
- https://github.com/cyclomon/UNSB
- https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image

## Data Sources

- Greek Sculptures (https://www.kaggle.com/datasets/thatgeeman/sculptures-of-greek-olympians-dataset)
- Human-FFHQ (https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?resource=download)
- Celeba_HQ (https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- Midjourney (https://www.midjourney.com/)
