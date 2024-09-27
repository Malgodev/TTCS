from argparse import ArgumentParser
import cv2
import numpy as np
from collections import deque
from mmaction.apis.inferencers import MMAction2Inferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input video device')
    parser.add_argument(
        '--vid-out-dir',
        type=str,
        default='',
        help='Output directory of videos.')
    parser.add_argument(
        '--rec',
        type=str,
        default=None,
        help='Pretrained action recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--label-file', type=str, default=None, help='label file for dataset.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the video in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--pred-out-file',
        type=str,
        default='',
        help='File to save the inference results.')

    call_args = vars(parser.parse_args())

    init_kws = ['rec', 'rec_weights', 'device', 'label_file']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    init_args["input_format"] = "array"
    mmaction2 = MMAction2Inferencer(**init_args)

    video_device = call_args.pop('inputs')
    cap = cv2.VideoCapture(video_device)
    sequence_len  = 5
    frames = deque(maxlen=sequence_len)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == sequence_len:
            inputs = np.array(frames)

            call_args["inputs"] = inputs
            results = mmaction2(**call_args)
            preds = results["predictions"][0]["rec_scores"][0]   
            # argmax of list
            pred_index = np.argmax(preds)

            cv2.putText(frame, pred_index, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()