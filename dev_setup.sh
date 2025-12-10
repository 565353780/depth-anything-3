cd ..
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git da3

cd da3
pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install xformers gradio pillow
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e .
