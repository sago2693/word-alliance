#!/usr/bin/env bash


# Create and activate the environment
if [[ "$OSTYPE" == "msys" ]]; then
    conda create -n dnlp python=3.8
    conda activate dnlp
else
    conda create -n dnlp python=3.8
    source activate dnlp
fi

conda create -n dnlp python=3.8
conda activate dnlp
conda install -c conda-forge jupyterlab
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7