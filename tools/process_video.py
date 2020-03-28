import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot,get_result
import argparse
from tqdm import tqdm
import os
import os.path as osp
import cv2
from cv2 import VideoWriter_fourcc
from pymongo import MongoClient
import logging
from utils import Sort
from utils.videoutils import video_tools as vt
import numba
import time
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
# 初始化人头追踪器
psn_tracker = Sort()
import asyncio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoints')
    parser.add_argument('--input_path',type=str,help='视频文件路径')
        
    parser.add_argument('--output_path',type=str,help='视频文件输出路径')
    parser.add_argument('--fps',type=int,default=30,help='输出的视频fps')
    parser.add_argument('--hat_color',type=str,default='green',help='安全帽框颜色')
    parser.add_argument('--person_color',type=str,default='red',help='人头框颜色')
    args = parser.parse_args()
    return args



    
    


def _get_logger(
    filename='log.txt'
    ,filemode="w"
    , format="%(asctime)s %(name)s:%(levelname)s:%(message)s"
    , datefmt="%d-%M-%Y %H:%M:%S"):

    logging.basicConfig(filename='log.txt',filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")
    return logging



def process_video(
    model,
    input_path,
    output_path,
    require_fps,
    hat_color,
    person_color,
    fourcc='mp4v'
    ):
    """处理视频并输出到指定目录
    
    Arguments:
        model {torch.nn.Sequ} -- [使用的模型]
        input_path {[str]} -- [视频文件路径]
        output_path {[str]} -- [视频文件输出路径]
        require_fps {[int]} -- [输出的视频fps]
        fourcc {[str]} -- [opencv写文件编码格式]
        hat_color {[str]} -- [安全帽框颜色]
        person_color {[str]} -- [人头框颜色]
        process_step {[int]} -- [以step分钟的间隔处理整个视频，内存越大step可以越大]
    """    
    video = mmcv.VideoReader(input_path)
    resolution = (video.width, video.height)
    video_fps = video.fps
    if require_fps is None:
        require_fps = video_fps
    if require_fps > video_fps:
        require_fps = video_fps
    vwriter = cv2.VideoWriter(
        output_path,
        VideoWriter_fourcc(*fourcc),
        require_fps,
        resolution
        )
    for frame in tqdm(video):
        result = inference_detector(model, frame)
        frame_result = get_result(
            frame,
            result,
            class_names=model.CLASSES,
            auto_thickness=True,
            color_dist={'hat':hat_color,'person':person_color})
        vwriter.write(frame_result)
    print('process finshed')


# def process_video1(
#     model,
#     input_path,
#     output_path,
#     require_fps,
#     hat_color,
#     person_color,
#     fourcc='mp4v',
#     process_step = 1
#     ):
#     """处理视频并输出到指定目录
    
#     Arguments:
#         model {torch.nn.Sequ} -- [使用的模型]
#         input_path {[str]} -- [视频文件路径]
#         output_path {[str]} -- [视频文件输出路径]
#         require_fps {[int]} -- [输出的视频fps]
#         fourcc {[str]} -- [opencv写文件编码格式]
#         hat_color {[str]} -- [安全帽框颜色]
#         person_color {[str]} -- [人头框颜色]
#         process_step {[int]} -- [以step分钟的间隔处理整个视频，内存越大step可以越大]
#     """    
#     assert isinstance(process_step,int),'处理步长必须为整数'

#     video = mmcv.VideoReader(input_path)
#     resolution = (video.width, video.height)
#     video_fps = video.fps
#     if require_fps is None:
#         require_fps = video_fps
#     if require_fps > video_fps:
#         require_fps = video_fps
#     vwriter = cv2.VideoWriter(
#         output_path,
#         VideoWriter_fourcc(*fourcc),
#         require_fps,
#         resolution
#         )
#     # 获取MongoDB数据库collection
#     collection = _get_collection()
#     # 获取子视频读取步长
#     read_step = video_fps * process_step * 60
#     # 将视频切分为子视频进行处理
#     start_index = 0
#     for over_index in range(read_step,len(video),read_step):
#         # 获取子视频片段　（加载进内存）
#         sub_video = video[start_index:over_index]
#         # 跟踪 
#         frames_indexs,results = vt.imgs_detection(sub_video,model)
#         # 获取探测到的没有戴好安全帽的bbox
#         bboxs_psn = results[:,1]
#         # 初始化跟踪器
#         psn_tracker = Sort()
#         psn_track_results = vt.track(bboxs_psn,psn_tracker)
#         # 插帧填充
#         famers_psn = 
#     for frame in tqdm(video):
#         result = inference_detector(model, frame)
#         frame_result = get_result(
#             frame,
#             result,
#             class_names=model.CLASSES,
#             auto_thickness=True,
#             color_dist={'hat':hat_color,'person':person_color})
#         vwriter.write(frame_result)
#     print('process finshed')



if __name__ == "__main__":
    args = parse_args()
    model = init_detector(args.config, args.checkpoints, device='cuda:0')
    if args.input_path is None:
        raise ValueError('input_path can not be None')
    if args.output_path is None:
        raise ValueError('output_path can not be None')
    process_video(
        model,
        args.input_path,
        args.output_path,
        args.fps,
        args.hat_color,
        args.person_color
    )
    
