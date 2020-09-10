#!/bin/bash
set -ex

export DEBIAN_FRONTEND=noninteractive
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
sudo apt install sox libsndfile1 ffmpeg wget -y
pip install unidecode wget frozendict wandb==0.9.7
pip install -e git://github.com/aljeshishe/nemo.git@93997a2c#egg=nemo[asr]
