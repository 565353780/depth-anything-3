pip install ninja pillow

pip install gradio==5.50.0

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

pip3 install -U xformers \
  --index-url https://download.pytorch.org/whl/cu124

pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
