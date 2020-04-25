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
from multiprocessing import Process,Pipe




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./detection/myconfigs/baseline/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--checkpoints',default='./fastr50_963.pth')
    args = parser.parse_args()
    return args

def conn_send(conn,imgs):
    conn.send(imgs)

def read_img(conn,fps=31):
    try:
        try:
            cap = VideoCaptureAsync(0)
        except:
            print("开启摄像头失败")
        # cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.start()
        sleep_time = 1/fps
        s = time.time()
        i = 1
        save_list = []
        while True:
            if len(save_list) == 3:
                conn.send(save_list)
                save_list = []
            _, img = cap.read()
            i+=1
            save_list.append(img)
            cv2.imshow("read_before",img)
            cv2.waitKey(int(sleep_time*1000-30))
            #time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        cap.stop()
        return



SHOW_IMG_Q = queue.Queue(maxsize=1000000)
READ_IMG_Q = queue.Queue(maxsize=1000000)

def pipline_read_img(conn,READ_IMG_Q):
    try:
        while True:
            #print('开始接受时间{}',format(time.time()-start))
            imgs = conn.recv()
            #print('结束接受时间{}',format(time.time()-start))
            for img in imgs:
                READ_IMG_Q.put(img)
    except KeyboardInterrupt:
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
                #print("len of show q {0}".format(SHOW_IMG_Q.qsize()))
    except KeyboardInterrupt:
        return


def show_img():
    try:
        s = time.time()
        i = 1
        while True:
            img = SHOW_IMG_Q.get()
            cv2.imshow("read_after",img)
            i+=1
            print("读取{}张图像耗时{}".format(i,time.time()-s))
            cv2.waitKey(10)
    except KeyboardInterrupt:
        return





if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    parent_conn, child_conn = Pipe()
    read = Process(target=read_img,args=(child_conn,))
    pipline_read = threading.Thread(target=pipline_read_img,args=(parent_conn,READ_IMG_Q))
    process = threading.Thread(target=process_img)
    show = threading.Thread(target=show_img)
    read.start()
    pipline_read.start()
    process.start()
    show.start()









