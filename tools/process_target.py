
import mmcv
from tqdm import tqdm
import os
import cv2
import numba
import math
import time
import pickle 
import argparse
import logging
import asyncio
import motor
import numpy as np
import os.path as osp
from utils import get_logger
from db import get_collection
from cv2 import VideoWriter_fourcc
from pymongo import MongoClient
from bson.binary import Binary 
from tracker import Tracker,track
from mmcv.visualization import color_val
from detection.mmdet.apis import init_detector, inference_detector, show_result_pyplot,get_result





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
        require_fps {[int]} -- [输出的视频fps]
        fourcc {[str]} -- [opencv写文件编码格式]
        hat_color {[str]} -- [安全帽框颜色]
        person_color {[str]} -- [人头框颜色]
        process_step {[int]} -- [以step分钟的间隔处理整个视频，内存越大step可以越大]
    """    
    video = mmcv.VideoReader(input_path)
    # 初始化人头追踪器
    psn_tracker = Tracker()
    resolution = (video.width, video.height)
    video_fps = video.fps
    ds = DetectionSifter(int(video_fps),args.input_path,1,3,resolution,get_collection())
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
        # bbox:(hat_bbox,person_bbox)
        bboxs = inference_detector(model, frame)
        frame_result = get_result(
                frame,
                bboxs,
                class_names=model.CLASSES,
                auto_thickness=True,
                color_dist={'hat':'green','person':'red'}
                )
        # person_bboxs:(N,5)
        person_bboxs = bboxs[1]
        # 筛选阈值大于0.5进行追踪
        person_bboxs = person_bboxs[person_bboxs[:,4] > 0.5]
        person_bboxs = np.expand_dims(person_bboxs,0)
        person_bboxs_tracks = track(person_bboxs,psn_tracker)[0]
        ds.add_object(person_bboxs_tracks,frame)
        vwriter.write(frame_result)
    ds.clear()
    print('process finshed')




class DetectionSifter(object):
    def __init__(
        self,
        fps,
        video_name,
        alive_thr,
        dead_thr,
        resolution,
        connection):
        assert isinstance(fps,int)
        assert isinstance(video_name,str)
        assert isinstance(alive_thr,(int,float))
        assert isinstance(dead_thr,(int,float))
        assert isinstance(resolution,tuple)
        # 正在处理的视频名字
        self.video_name = video_name
        # 用于缓存探测到的bbox
        self.buffer = {}
        # 处理的视频的fps
        self.fps = fps
        # 处理的帧数
        self.process_step = 0
        # 数据库连接
        self.conn = connection
        # 生存时间阈值（一个目标被跟踪持续了多久才认为它的确是被探测到了）
        self.alive_thr = alive_thr
        # 死亡清除时间　(一个object死亡多久才会去清除))
        self.dead_thr = dead_thr
        # 视频图像的正中心坐标，用于计算最佳保存bbox (W,H)
        self.center = (resolution[0] // 2,resolution[1] // 2)
        # 矩形框粗细大小
        self.thickness = (resolution[0] + resolution[1]) // 600
    def add_object(self,bboxs,img):
        assert isinstance(bboxs,np.ndarray),'bbox is type is {0}'.format(type(bboxs))
        assert bboxs.shape[1] == 5
        self.process_step += 1
        for bbox in bboxs:
            # 取出id
            id_num = int(bbox[4])
            # 取出坐标
            bbox = bbox[:-1] # shape (4,)
            bbox = np.expand_dims(bbox,0) # shape (1,4)
            bbox = bbox.astype(int) # 转换为int类型
            if id_num not in self.buffer.keys():
                self.buffer[id_num] = {
                    'bbox':bbox, # shape (1,4) 包含有该目标从进入到消失所有运动的框移动数据
                    # 记录该目标被跟踪开始的时间(视频时间)
                    'start_time':self.process_step / self.fps,
                    # 记录该目标失去跟踪的时间
                    'over_time':self.process_step / self.fps,
                    # 记录该目标被跟踪开始所处的帧数
                    'start_step':self.process_step,
                    # 记录该目标失去跟踪所处的帧数 
                    'over_step':self.process_step,
                    # 用来记录哪个bbox是最佳的证据图　根据与中心的欧氏距离来判定
                    'best_bbox_index':0,
                    # 记录最佳的证据图所处的视频时间
                    'best_img_time':self.process_step / self.fps,
                    'img':self._draw_bbox(img,bbox),
                    # 记录该目标是什么时候(标准时间,视频时间)被侦测到的
                    'img_save_time':(self._get_time(),self.process_step / self.fps)
                }
            else:
                x = self.buffer[id_num]
                # shape (n,4) n为存活的帧数
                x['bbox'] = np.concatenate((x['bbox'],bbox),axis=0)
                x['over_time'] = self.process_step / self.fps
                x['over_step'] = self.process_step
                bbi = x['best_bbox_index']
                # 如果新跟踪到的矩形框要比旧的最佳矩形框更接近中心，便认为此矩形框所在的图像更
                # 适合作为新的截图证据
                if self._distance2center(bbox[0]) > self._distance2center(x['bbox'][bbi]):
                    # 刷新最佳bbox下标
                    x['best_bbox_index'] = len(x['bbox']) - 1
                    # 刷新保存的证据图像
                    x['img'] = self._draw_bbox(img,bbox)
                    # 刷新最佳图像保存时间
                    x['img_save_time'] = (self._get_time(),self.process_step / self.fps)

        # 每隔三秒钟检测整个缓冲区，看是否需要保存或者清除一些object
        if (self.process_step / self.fps) % 3 == 0:
            self._check_all_object()
    
    def _get_alive_time(self,detec_object):
        # 计算指定的object存活时间
        return len(detec_object['bbox']) / self.fps


    def _distance2center(self,bbox):
        # 计算指定的bbox到center的距离 
        x1,y1,x2,y2 = bbox
        bbox_center = (x2-x1,y2-y1)
        return math.sqrt(
            (self.center[0] - bbox_center[0])**2 + (self.center[1] - bbox_center[1])**2
        )

    def _check_all_object(self,is_last = False):
        if is_last:
            self.dead_thr = 0
        Loger.info('检查'+str(self.buffer.keys()))
        # 检查缓存区的内容
        for id_num in self.buffer.copy():
            detec_object = self.buffer[id_num]
            # 获取探测到的目标的死亡时间(也就是跟踪的目标失去跟踪的时间)
            dead_time = self.process_step / self.fps - detec_object['over_time']
            # 如果死亡时间大于 self.dead_thr 秒,则可以认为该目标已经不会再出现在序列中
            if dead_time >= self.dead_thr:
                # 开始检查该目标所持续的时间(存活时间)
                live_time = self._get_alive_time(detec_object)
                Loger.info('死亡时间大于３秒'+str(id_num))
                if live_time > self.alive_thr:
                    Loger.info('{0}存活时间大于{1}秒'.format(str(id_num),self.alive_thr)+str(self.buffer.keys()))
                    # TODO:数据库保存
                    # 删除该字典元素
                    Loger.info('取出{0}放入数据库'.format(str(id_num)))
                    doc = {
                            # 现实时间，精确到秒
                            'time':detec_object['img_save_time'][0],
                            'info':{
                            # 图像二进制数据，需要用pick
                            'img':Binary(pickle.dumps(detec_object['img'])),
                            'img_shape':detec_object['img'].shape,
                            'video_name':self.video_name,
                            'video_time':detec_object['img_save_time'][1]
                            }
                    }
                    self.conn.insert_one(doc)
                    Loger.info('Add a new record to db {0}'.format(doc['info']['img_shape']))
                    # # 异步循环器
                    # loop = asyncio.get_event_loop()
                    # # 定义异步插入
                    # async def do_insert():
                    #     result = await self.conn.insert_one({
                    #         'time':detec_object['img_save_time'][0],
                    #         'info':{
                    #         'img':detec_object['img'],
                    #         'video_name':self.video_name,
                    #         'video_time':detec_object['img_save_time'][1]
                    #         }
                    #     })
                    #     Loger.info('Add a new record to db {0}'.format(result.inserted_id))
                    # loop.run_until_complete(do_insert())
                    del self.buffer[id_num]
                else:
                    Loger.info('{0}存活时间小于{1}秒'.format(str(id_num),self.alive_thr)+str(self.buffer.keys()))
                    Loger.info('删除{0}'.format(str(id_num)))
                    # 删除该字典元素
                    del self.buffer[id_num]
                    # 抬走下一个
                    continue

    def _draw_bbox(self,img,bbox):
        bbox = bbox[0]
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(
            img,left_top,right_bottom, color_val('red'), thickness=self.thickness
        )
        return img


    def _get_time(self):
        # 获取当前时间字符串
        ct  = time.localtime(time.time())
        current_time = '{0}-{1}-{2} {3}:{4}:{5}'.format(
            ct.tm_year,ct.tm_mon,ct.tm_mday,ct.tm_hour,ct.tm_min,ct.tm_sec)
        return current_time

    def clear(self):
        # 清理剩余的数据
        self._check_all_object(True)


if __name__ == "__main__":
    args = parse_args()
    if args.input_path is None:
        raise ValueError('input_path can not be None')
    model = init_detector(args.config, args.checkpoints, device='cuda:0')
    Loger = get_logger()
    process_video(
        model,
        args.input_path,
        args.output_path,
        args.fps,
        args.hat_color,
        args.person_color
    )