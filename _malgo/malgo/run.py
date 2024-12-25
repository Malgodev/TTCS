from mmaction.apis import inference_recognizer, init_recognizer
from mmengine.registry import MODELS
from mmengine import Config
import torch
from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

# Path to your configuration file
config_file = '_malgo/malgo/tsn_malgo-pretrained-t18.py'  # Replace with your actual config path
# Path to your trained checkpoint
checkpoint_path = 'work_dirs/guide/malgo_4167.pth'  # Replace with your .pth file

# Load the configuration
cfg = Config.fromfile(config_file)

# Initialize the recognizer model
# `init_recognizer` handles model building and checkpoint loading for you
model = init_recognizer(cfg, checkpoint=checkpoint_path, device='cuda')

# Path to the video you want to test
video_path = 'data/malgo/train/B_A-1_2.mp4'

# Path to your label map
label_map_path = 'data/malgo/malgo_label.txt'

# Load label map
with open(label_map_path, 'r') as f:
    label_map = [line.strip() for line in f.readlines()]

# Perform inference
results = inference_recognizer(model, video_path)

# Get top-k predictions
top_k = 5
scores = results[0]
top_indices = scores.argsort()[-top_k:][::-1]

print("Top Action Classifications:")
for idx in top_indices:
    print(f"{label_map[idx]}: {scores[idx]:.4f}")
