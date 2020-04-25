import argparse
import os.path as osp
import os
from datetime import *
import time
import cv2
from cv2 import VideoWriter_fourcc
from utils.rtmpWriter import RtmpStreamWriter
from multiprocessing import Process,Pipe
import subprocess as sp
import numpy as np
import queue
import threading
from tqdm import tqdm
import mmcv
from utils import get_logger,ImageHandleProcess
from utils.loginFile import Login
from utils.neuralNetworkModelManager import *
import  json
from detection import init_detector

from tracker import Tracker,track,over_step_video,fill,draw_label,imgs_detection,img_detection,DetectionSifter


def test():
    print("创建进程")
"""
监控管理器，采用单例模式，该类用于操控监控摄像头，并且将摄像头拍摄的内容进行计算，存储到数据，发送给前端显示
"""
class MonitorHandler(object):
    _flag = True
    _isOpen=False
    _instant=None
    _imageQueue_capture=queue.Queue(10000)        #摄像头将拍摄到的图片存入到该队列中
    _imageQueue_writeRtmp=queue.Queue(10000)       #存放待写成rtmp流的图片
    _Img_Q=queue.Queue(10000)               #存放待写成视频的图片
    _captureFps=None                #摄像头的帧数
    _resolution=None                        #摄像头分辨率
    _step=3                         #跳步数,默认为3
    _speed=0                        #处理速度
    _handleCount_peerSecond=0     #每秒钟处理的帧数，用于计算速度
    _captureUrl=None
    _rtmpUrl=None
    _password="monitorHandler"
    _neuralNetworkModel=None    #神经网络模型管理器
    _pipeReader=None
    _pipeWriter=None
    def __new__(cls, *args, **kwargs):
        if cls._instant is None:
            cls._instant = super().__new__(cls)
        return cls._instant
    def __init__(self):
        if not MonitorHandler._flag:
            return
        MonitorHandler._flag = False

    """
    设置摄像头url，和rtmpurl
    """
    def setUrl(self,captureUrl,rtmpUrl):
        self._captureUrl=captureUrl
        self._rtmpUrl=rtmpUrl
    """
    摄像头是否已经打开
    """
    def isOpen(self):
        return self._isOpen

    """
    获取视频处理速度
    """
    def getHandleSpeed(self):
        return self._speed
    """
    外部接口,打开摄像头,处理摄像头发送过来的视频流,打开成功则返回True,否则返回False
    """
    def openMonitor(self):
        #检测摄像头是否已经打开
        if self._isOpen:
            return True
        #检查是否设置了URL
        if self._rtmpUrl is  None or self._captureUrl is  None:
            return False
        #获取神经网络模型管理器
        self._neuralNetworkModel=NeuralNetworkModelManager()
        #尝试占用神经网络模型
        flag=self._neuralNetworkModel.occupyModel(self._password)
        #如果神经网络模型已经被占用，则返回False
        if not flag:
            return False
        #创建摄像头对象
        try:
            cap = cv2.VideoCapture(self._captureUrl)
        except:
            return False
        if cap==None:
            return False
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS)

        rtmpWriter=RtmpStreamWriter(width,height,fps,self._rtmpUrl)
        # 获取摄像头的分辨率
        self._resolution = (width, height)
        print("摄像头分辨率：{0}".format(self._resolution))
        #获取摄像头的帧数
        self._captureFps=(fps)
        #创建照片捕捉线程

        self._pipeReader, self._pipeWriter = Pipe(True)
        thread_captrueImage = threading.Thread(target=self._openVideoCapture, args=(cap,))
        #process_captureImage = Process(target=self._openVideoCapture,args=(cap,self._pipeWriter,))

        #创建视频流处理线程
        thread_handleVideoStream=threading.Thread(target=self._handleVideoStream,args=(rtmpWriter,))
        thread_captrueImage.start()
        #process_captureImage.start()
        #print("进程ＩＤ：{0}".format(process_captureImage.pid))
        thread_handleVideoStream.start()
        self._isOpen=True;
        return True





    """
    内部接口，打开摄像头,录制视频,将拍摄的视频以图片的形式存入图片缓冲队列里面,多线程调用
    """
    def _openVideoCapture(self,cap):
        cv2.namedWindow('djf', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        #摄像机的帧数
        captureFps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            # 如果摄像头关闭，则退出
            # if not self._isOpen:
            #     #释放神经网络模型模型
            #     self._neuralNetworkModel.releaseModel(self._password)
            #     break
            #获取摄像头拍摄的图片
            ret, frame = cap.read()
            cv2.imshow("djf", frame)
            self._imageQueue_capture.put_nowait(frame)
            #pipeWriter.send(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._isOpen=False
                # 释放神经网络模型模型
                self._neuralNetworkModel.releaseModel(self._password)
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    """
    内部接口,处理视频流(图片缓冲队列),多线程调用
    """
    def _handleVideoStream(self, rtmpWriter, fourcc='avc1'):
        dateStr=datetime.now().strftime("%Y-%m-%d")
        count=self.__getMonitorVideoCount_InOneDay(dateStr)
        relPath="./monitorVideo/{0}_{1}.mp4".format(dateStr,count)
        outputPath=osp.abspath(relPath)
        # 探测过滤器,将神经网络处理出的信息进行不间断处理,并将其中违规图像写入数据库
        ds = DetectionSifter(
            int(self._captureFps),
            osp.basename(outputPath).split('.')[0],
            self._resolution,
            Login('data', 'picAboutNotWearHat')
        )
        # 用于将图像存储为视频
        vwriter = cv2.VideoWriter(
            outputPath,
            VideoWriter_fourcc(*fourcc),
            self._captureFps,
            self._resolution
        )
        p = ImageHandleProcess(self._neuralNetworkModel.getModel(self._password), self._step + 1)

        #开启写视频线程
        thread_writeVideo = threading.Thread(target=self._write_frame, args=(vwriter, rtmpWriter,))
        thread_writeVideo.start()
        #开启测速线程
        thread_calculateSpeed=threading.Thread(target=self._calculateHandleSpeed)
        thread_calculateSpeed.start()

        startImg=None
        startIndex=0
        myTime=datetime.now()
        #开始循环处理视频流
        while True:
            frameList=[]
            #将上一批图片中最后一张为下一批的第一张图片
            if startImg is not None:
                frameList.append(startImg)
            nowStep=self._step
            #根据跳步数目，创建图片列表
            while len(frameList) != nowStep+1:
                try:
                    frameList.append(self._pipeReader.recv())
                    #frameList.append(self._imageQueue_capture.get(timeout=3))
                except:
                    #检查摄像头是否已经关闭，如果关闭了，则退出线程
                    if not self._isOpen:
                        return
            startImg=frameList[-1]
            origin_frames = np.array(frameList)
            #
            frames_index = [startIndex, startIndex+nowStep]
            startIndex=startIndex+nowStep
            startTime=datetime.now()
            origin_frames, psn_objects, hat_objects = p.get_img(origin_frames, frames_index)
            endTime=datetime.now()
            #print("开始时间:{0},结束时间：{1}".format(startTime,endTime))
            for psn_o in psn_objects:
                ds.add_object(*psn_o)
            self._Img_Q.put(origin_frames[:-1])
            self._handleCount_peerSecond+=nowStep
            if(self._handleCount_peerSecond>=30):
                print("耗时：{0}".format((datetime.now()-myTime)))
                #print("计算{0}张图片,开始时间：{1},结束时间{2}".format(self._handleCount_peerSecond,myTime,datetime.now()))
                myTime=datetime.now()
                self._handleCount_peerSecond=0
        ds.clear()

    """
    内部接口，①将图片写成视频文件，②将图片写成rtmp流送到srs服务器上,在前端实时显示。多线程调用
    """
    def _write_frame(self, vwriter, rtmpWriter):
        count =0
        while True:
            try:
                imgs = self._Img_Q.get(timeout=3)
                for img in imgs:
                    count+=1
                    vwriter.write(img)
                    rtmpWriter.write(img)
            except:
                #检查摄像头是否仍然开启，否则退出线程
                if not self._isOpen:
                    return


    """
    内部方法，获取当天监控录像的数量:
    """
    def __getMonitorVideoCount_InOneDay(self,videoName_noFormat):
        videoDir = os.path.abspath('./monitorVideo/')
        count = 0
        for root, dirs, files in os.walk(videoDir):
            for file in files:
                if file.split('_')[0] == videoName_noFormat:
                    count += 1
        return count

    """
    内部方法，计算处理速度,多线程调用
    """
    def _calculateHandleSpeed(self):
        while self._isOpen:
            #print("睡前：{0}".format(datetime.now()))
            time.sleep(1)
            #print("睡后：{0}".format(datetime.now()))
            #print("数量：{0}".format(self._handleCount_peerSecond))
            #self._speed=self._handleCount_peerSecond;
            #self._handleCount_peerSecond=0
            #print("当前处理速度为{0}".format(self._speed))

