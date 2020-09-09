#!/bin/bash
set -ex

export DEBIAN_FRONTEND=noninteractive
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
apt install sox libsndfile1 ffmpeg wget -y
pip install nemo_toolkit[asr]==0.11.0 unidecode wget frozendict kaldi_io torch-stft==0.1.4 soundfile==0.10.3.post1

mkdir -p configs
wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/master/examples/asr/notebooks/configs/jasper_an4.yaml