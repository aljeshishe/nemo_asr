pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
apt-get install sox libsndfile1 ffmpeg
pip install nemo_toolkit unidecode wget

mkdir configs
wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/master/examples/asr/notebooks/configs/jasper_an4.yaml