# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import torch
import math

from mmdet.apis import inference_detector, init_detector


classNames = ["person"]
treshold = 0.5
rec_size = (50, 50)
rec_color = (67, 88, 168) #(0, 0, 255)
mouseX, mouseY = 0, 0


def draw_rectangle(image):
    global mouseX, mouseY
    alfa = mouseX/400
    color = tuple(i*alfa+j*(1-alfa) for i, j in zip(rec_color, (0, 0, 255)))
    start_point = mouseX, mouseY
    end_point = (start_point[0] + rec_size[0], start_point[1] + rec_size[1])
    return cv2.rectangle(image, start_point, end_point, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def draw_results(img, results):
    for r in results.pred_instances:
        for box, label, score in zip(r.bboxes, r.labels, r.scores):
            # bounding box
            if score < treshold:
                continue
            
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((score*100))/100
            print("Confidence --->",confidence)

            # class name
            print("Class name -->", classNames[label])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[label], org, font, fontScale, color, thickness)
    return img


def set_coords(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y


def main():
    draw = False
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)
 
    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        
        if draw:
            img = draw_rectangle(img)

        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        draw = ch == ord('d') or ch == ord('D')
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        img = draw_results(img, result)
        cv2.imshow('Webcam', img)
        cv2.setMouseCallback('Webcam', set_coords)


if __name__ == '__main__':
    main()