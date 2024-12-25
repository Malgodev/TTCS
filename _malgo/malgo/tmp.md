## Webcam demo

We provide a demo script to implement real-time action recognition from web camera. In order to get predict results in range `[0, 1]`, make sure to set `model.cls_head.average_clips='prob'` in config file.

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--camera-id ${CAMERA_ID}] [--threshold ${THRESHOLD}] \
    [--average-size ${AVERAGE_SIZE}] [--drawing-fps ${DRAWING_FPS}] [--inference-fps ${INFERENCE_FPS}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `CAMERA_ID`: ID of camera device If not specified, it will be set to 0.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.
- `AVERAGE_SIZE`: Number of latest clips to be averaged for prediction. If not specified, it will be set to 1.
- `DRAWING_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 20.
- `INFERENCE_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 4.

If your hardware is good enough, increasing the value of `DRAWING_FPS` and `INFERENCE_FPS` will get a better experience.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
   and outputting result labels with score higher than 0.2.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth tools/data/kinetics/label_map_k400.txt --average-size 5 \
     --threshold 0.2 --device cpu
   ```

2. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
     tools/data/kinetics/label_map_k400.txt --average-size 5 --threshold 0.2 --device cpu
   ```

3. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
   and outputting result labels with score higher than 0.2.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/i3d_r50_32x2x1_video_infer.py \
     checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth tools/data/kinetics/label_map_k400.txt \
     --average-size 5 --threshold 0.2
   ```

Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case.
Users can change:

- `SampleFrames` step (especially the number of `clip_len` and `num_clips`) of `test_pipeline` in the config file, like `--cfg-options test_pipeline.0.num_clips=3`.
- Change to the suitable Crop methods like `TenCrop`, `ThreeCrop`, `CenterCrop`, etc. in `test_pipeline` of the config file, like `--cfg-options test_pipeline.4.type=CenterCrop`.
- Change the number of `--average-size`. The smaller, the faster.
