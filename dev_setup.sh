cd ..
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git da3
#git clone https://github.com/nerfstudio-project/gsplat.git
git clone --recursive https://github.com/facebookresearch/xformers.git

pip install ninja pillow

pip install gradio==5.50.0

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

cd xformers
git checkout v0.0.33
python setup.py install

pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

cd ../da3
pip install -e .
