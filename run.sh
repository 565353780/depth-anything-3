cd ../da3

CUDA_VISIBLE_DEVICES=7 \
  python src/depth_anything_3/app/gradio_app.py \
  --host 0.0.0.0 \
  --port 7861 \
  --model-dir $HOME/chLi/Model/DepthAnythingV3/DA3-GIANT/
