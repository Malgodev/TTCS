import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type

import keyboard

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)
MSGCOLOR = (128, 128, 128)
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Game action recognition')

    # Các trường bắt buộc
    parser.add_argument('config', help='đường dẫn đến file config')
    parser.add_argument('checkpoint', help='đường dẫn đến file checkpoint (file có đuôi .pth)')
    parser.add_argument('label', help='đường dẫn đến file label')

    # Các trường k bắt buộc
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA hoặc cpu')
    parser.add_argument(
        '--threshold', type=float, default=0.01, help='ngưỡng thấp nhất để hiện thị')
    parser.add_argument(
        '--average-size', type=int, default=1, help='Số lượng frame được lưu vào cache để dự đoán')
    parser.add_argument(
        '--drawing-fps', type=int, default=20, help='Đặt giới hạn cho thời gian vẽ, có thể hiểu là fps của webcam nhận')
    parser.add_argument('--inference-fps', type=int, default=4, help='Đặt số lượng frame mà model nhận diện hành động')
    
    args = parser.parse_args()

    assert args.drawing_fps >= 0 and args.inference_fps >= 0, 'FPS phải lớn hơn 0, hoặc 0 để bỏ giới hạn'

    return args

def active_webcam():
    print("Nhấn Esc để thoát")

    text_info = {}
    cur_time = time.time()

    while True:
        msg = 'Waiting for action ...'

        _, frame = camera.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result

                if score < threshold:
                    break
                
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score * 100, 2))

                text_info[location] = text

                cv2.putText(frame, text, location, 
                            FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
                

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location,
                            FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
        
        else:
            cv2.putText(frame, msg, (0, 40), 
                       FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("ok")
            camera.release()
            cv2.destroyAllWindows()
            break

        if drawing_fps > 0:
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()

    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))

                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        # Set các giá trị vào cur data
        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        # 
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        scores = result.pred_score.tolist()

        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            
            results = score_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

            if inference_fps > 0:
                sleep_time = 1 / inference_fps - (time.time() - cur_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cur_time = time.time()


def main():
    global average_size, threshold, drawing_fps, inference_fps
    global device, model, camera, data, label, sample_length
    global test_pipeline, frame_queue, result_queue

    args = parse_args()
    print(args)

    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)

    #  Khởi tạo recognizer
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    camera = cv2.VideoCapture(0)
    data = dict(img_shape=None, modality='RGB', label=-1)

    # Load label từ label file
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # test pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    # TODO rewrite pipeline
    pipeline_ = pipeline.copy()

    print(pipeline)

    for step in pipeline:
        print(step)

        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            # Xoá bước sample frame bởi webcam tự tạo frame rồi.
            pipeline_.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            pipeline_.remove(step)

    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)

        pw = Thread(target=active_webcam, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)

        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
