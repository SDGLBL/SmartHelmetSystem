import mmcv
from tqdm import tqdm
import os
import cv2
import math
import time
import pickle
import argparse
import datetime
import logging
import queue
import threading
import numpy as np
import os.path as osp
from utils import get_logger,ImageHandleProcess
from db import get_collection
#from cv2 import VideoWriter_fourcc
from pymongo import MongoClient
from bson.binary import Binary
from tracker import Tracker,track,over_step_video,fill,draw_label,imgs_detection,img_detection,DetectionSifter
from mmcv.visualization import color_val
from detection.mmdet.apis import init_detector, inference_detector, show_result_pyplot,get_result
from utils.loginFile import Login
from utils.async_cv import VideoCaptureAsync

READ_IMG_Q = queue.Queue(maxsize=1000000)
SHOW_IMG_Q = queue.Queue(maxsize=1000000)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./detection/myconfigs/baseline/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--checkpoints',default='./fastr50_963.pth')
    args = parser.parse_args()
    return args

def read_img(fps=30):
    try:
        try:
            cap = VideoCaptureAsync(0)
        except:
            print("开启摄像头失败")
        cap.start()
        sleep_time = 1/fps
        while True:
            _, img = cap.read()
            READ_IMG_Q.put(img)
            time.sleep(sleep_time)
            print("len of read q {0}".format(READ_IMG_Q.qsize()))
    except KeyboardInterrupt:
        cap.stop()
        return

def process_img(step = 6):
    try:
        model = init_detector(args.config, args.checkpoints, device='cuda')
        LAST_IMG = READ_IMG_Q.get()
        p = ImageHandleProcess(model, step)
        while True:
            IMG_LIST = []
            for i in range(step):
                if i == 0:
                    IMG_LIST.append(LAST_IMG)
                elif i == step - 1:
                    img = READ_IMG_Q.get()
                    IMG_LIST.append(img)
                    LAST_IMG = img
                else:
                    img = READ_IMG_Q.get()
                    IMG_LIST.append(img)
            frames_index = [0, step-1]
            origin_frames, psn_objects, hat_objects = p.get_img(IMG_LIST, frames_index)
            for frame in origin_frames[:-1]:
                SHOW_IMG_Q.put(frame)
                print("len of show q {0}".format(SHOW_IMG_Q.qsize()))
    except KeyboardInterrupt:
        return


def show_img():
    try:
        while True:
            img = SHOW_IMG_Q.get()
            cv2.imshow("test",img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        return





if __name__ == '__main__':
    args = parse_args()
    read = threading.Thread(target=read_img)
    process = threading.Thread(target=process_img)
    show = threading.Thread(target=show_img)
    read.start()
    process.start()
    show.start()









