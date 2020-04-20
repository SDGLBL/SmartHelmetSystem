import argparse
import os.path as osp
import os
from datetime import *
from time import *
import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
import queue
import threading
from tqdm import tqdm
import mmcv
from utils import get_logger,Process

from utils.loginFile import Login
from utils.neuralNetworkModel import *

from detection import init_detector

from tracker import Tracker,track,over_step_video,fill,draw_label,imgs_detection,img_detection,DetectionSifter

"""
该类用于处理视频
"""
class VideoHandler(object):
    _instant = None
    _videoList=[]   #任务列表，里面存放视频的名字
    _isWork=False   #是否正在工作
    _outPutFps=30;
    _hatColor="green"
    _personColor="red"
    flag = True
    def __new__(cls, *args, **kwargs):
        if cls._instant is None:
            cls._instant = super().__new__(cls)
        return cls._instant
    def __init__(self):
        if not VideoHandler.flag:
            return
        VideoHandler.flag = False
    """
    该方法为添加一个任务到任务列表里面
    """
    def addVideo(self,fileName):
        self._videoList.append(fileName)

    """
    当前是否正在执行任务
    """
    def isWork(self):
        return self._isWork
    """
    获取当前处理的视频名字,如果当前没有任务，则返回None
    """
    def getCurrentVideoName(self):
        if self._isWork:
            return self._videoList[0]
        else:
            return None
    """
    获取当前任务执行状态,返回两个结果,第一个为任务的总时长,第二个为完成任务所需要的剩余时间,如果没有任务,则返回none
    """
    def getCurrentVideoState(self):
        return None
    """
    外部接口开始处理视频,如果任务开始或已经正在进行则返回true,如果任务数量为0则返回false
    """
    def startHandleVideo(self):
        if len(self._videoList)>0:
            #判断任务是否正在进行，如果正在进行则返回true,如果没有进行，则再判断神经网络模型是否被占用
            if not self._isWork:
                #创建线程
                t=threading.Thread(target=self._handleVideoList)
                t.start()
                self._isWork=True
            return True
        else:
            return False
    """
    内部接口,循环处理视频队列里面的视频，该方法用多线程启用
    """
    def _handleVideoList(self):
        #获取神经网络模型
        neuralNetworkModel=NeuralNetworkModel()
        model=neuralNetworkModel.getModel()
        #判断神经网络是否被占用,如果被占用则返回
        if model==None:
            self._isWork=False
            return
        while True:

            #如果任务列表已经为空，则退出线程
            if len(self._videoList)==0:
                break
            #取出任务列表的第一个任务
            videoName=self._videoList[0]
            #视频的输入路径
            inputPath=os.path.abspath('./static/uploads/'+videoName)
            videoName_noFormat=videoName.split('.')[0]
            #获取数据库中与该视频是同一天视频的数量
            videoCount_InOneDay=self._getVideoCount_InOneDay(videoName_noFormat)
            #视频的输出路径
            outputPath=os.path.abspath("./static/video/{0}_{1}.mp4".format(videoName_noFormat,videoCount_InOneDay))
            print ("视频输入路径：{0}".format(inputPath))
            print("视频输出路径：{0}".format(outputPath))
            #开始处理视频
            self._process_video(
                model,
                0.5,
                inputPath,
                outputPath,
                self._outPutFps,
                self._hatColor,
                self._personColor
                )
            #将完成的任务删除
            self._videoList.pop(0)
        #设置工作状态为False
        self._isWork=False
        #释放神经网络模型
        neuralNetworkModel.releaseModel()
    """
    内部接口,处理视频
    """
    def _process_video(self,
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
        startTime = datetime.now()
        #创建图像缓冲队列
        Img_Q = queue.Queue(100)
        video = mmcv.VideoReader(input_path, cache_capacity=40)
        # 图像分辨率
        resolution = (video.width, video.height)
        video_fps = video.fps
        # 探测过滤器,将神经网络处理出的信息进行不间断处理,并将其中违规图像写入数据库
        ds = DetectionSifter(
            int(video_fps),
            osp.basename(output_path).split('.')[0],
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
        t = threading.Thread(target=self._write_frame, args=(vwriter,Img_Q,))
        t.start()
        # 　跳步视频（每次读取step+1帧）
        # video = over_step_video(video,step)
        vlen = len(video)

        indexs = np.arange(vlen)[0:len(video):step]
        # 每次取step张图像，比如第一次取[0:3]第二次则取[2:5]确保探测一张跳过step-1张,相当于探测一张相当于探测step张
        p = Process(model, step + 1)
        for start_index in tqdm(indexs):
            end_index = start_index + step + 1
            if end_index >= vlen:
                break
            origin_frames = video[start_index:end_index]
            origin_frames = np.array(origin_frames)
            #
            frames_index = [start_index, start_index + step]
            origin_frames, psn_objects, hat_objects = p.get_img(origin_frames, frames_index)
            for psn_o in psn_objects:
                ds.add_object(*psn_o)
            Img_Q.put(origin_frames[:-1])
        ds.clear()
        print('process finshed')
        endTime=datetime.now()
        print("开始时间:{0},结束时间:{1}".format(startTime,endTime))
        print("总共耗时：{0}".format((endTime-startTime).seconds))
    """
    内部接口,获取数据库中，某天的视频的数量
    """
    def _getVideoCount_InOneDay(self,videoName_noFormat):
        videoDir=os.path.abspath('../static/video/')
        count=0
        for root, dirs, files in os.walk(videoDir):
            for file in files:
                if file.split('.')[0]==videoName_noFormat:
                    count+=1
        return count
    """
    内部接口，将图片队列里面的图片写入到视频文件中，多线程调用
    """
    def _write_frame(self,vwriter,Img_Q):
        count =0
        while True:
            try:
                imgs = Img_Q.get(timeout=5)
                for img in imgs:
                    count+=1
                    vwriter.write(img)
            except:
                print('写入结束总写入图片：{0}'.format(count))
                return

