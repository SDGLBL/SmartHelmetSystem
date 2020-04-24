
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
from utils import get_logger,Process
from db import get_collection
from cv2 import VideoWriter_fourcc
from pymongo import MongoClient
from bson.binary import Binary 
from tracker import Tracker,track,over_step_video,fill,draw_label,imgs_detection,img_detection,DetectionSifter
from mmcv.visualization import color_val
from detection.mmdet.apis import init_detector, inference_detector, show_result_pyplot,get_result
from utils.loginFile import Login






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./detection/myconfigs/baseline/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--checkpoints',default='./fastr50_963.pth')
    parser.add_argument('--i',type=str,help='视频文件路径',default='./test.mp4')
    parser.add_argument('--thre',type=float,help='目标阈值',default=0.5)
    parser.add_argument('--o',type=str,help='视频文件输出路径',default='./test_out.mp4')
    parser.add_argument('--fps',type=int,default=30,help='输出的视频fps')
    parser.add_argument('--hat_color',type=str,default='green',help='安全帽框颜色')
    parser.add_argument('--person_color',type=str,default='red',help='人头框颜色')
    args = parser.parse_args()
    return args

def write_frame(vwriter):
    while True:
        try:
            imgs = Img_Q.get(timeout=5)
            for img in imgs:
                vwriter.write(img)
        except:
            print('写入结束')
            return


def process_video(
    model,
    thre,
    input_path,
    output_path,
    require_fps,
    hat_color,
    person_color,
    fourcc='avc1',
    step=2,
    ):
    """处理视频并输出到指定目录
    
    Arguments:
        model {torch.nn.Sequ} -- [使用的模型]
        input_path {[str]} -- [视频文件路径]
        require_fps {[int]} -- [输出的视频fps]
        fourcc {[str]} -- [opencv写文件编码格式]
        hat_color {[str]} -- [安全帽框颜色]
        person_color {[str]} -- [人头框颜色]
    """    
    video = mmcv.VideoReader(input_path,cache_capacity=40)
    # 图像分辨率
    resolution = (video.width, video.height)
    video_fps = video.fps
    # 探测过滤器,将神经网络处理出的信息进行不间断处理,并将其中违规图像写入数据库
    ds = DetectionSifter(
        int(video_fps),
        osp.basename(args.i).split('.')[0],
        resolution,
        Login('data', 'picAboutNotWearHat')
        )
    if require_fps is None:
        require_fps = video_fps
    if require_fps > video_fps:
        require_fps = video_fps
    # 用于将图像存储为视频
    vwriter = cv2.VideoWriter(
        output_path,
        VideoWriter_fourcc(*fourcc),
        require_fps,
        resolution
        )
    t = threading.Thread(target=write_frame,args=(vwriter,))
    t.start()
    #　跳步视频（每次读取step+1帧）
    #video = over_step_video(video,step)
    vlen = len(video)

    indexs = np.arange(vlen)[0:len(video):step]
    # 每次取step张图像，比如第一次取[0:3]第二次则取[2:5]确保探测一张跳过step-1张,相当于探测一张相当于探测step张
    p = Process(model,step+1)
    for start_index in tqdm(indexs):
        end_index = start_index + step + 1
        if end_index >= vlen:
            break
        origin_frames = video[start_index:end_index]
        origin_frames = np.array(origin_frames)
        # 
        frames_index = [start_index,start_index+step]
        origin_frames,psn_objects,hat_objects = p.get_img(origin_frames,frames_index)
        for psn_o in psn_objects:
            ds.add_object(*psn_o)
        Img_Q.put(origin_frames[:-1])
    ds.clear()
    print('process finshed')



if __name__ == "__main__":
    args = parse_args()
    if args.i is None:
        raise ValueError('input_path can not be None')
    model = init_detector(args.config, args.checkpoints, device='cuda:0')
    Img_Q = queue.Queue(100)
    process_video(
        model,
        args.thre,
        args.i,
        args.o,
        args.fps,
        args.hat_color,
        args.person_color
    )