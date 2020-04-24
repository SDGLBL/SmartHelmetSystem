import numpy as np
from detection.mmdet.apis import inference_detector
import cv2
from . import Tracker,track
from numba import jit
from mmcv import VideoReader
import numpy as np
import time,math
from bson.binary import Binary 
from db import get_collection
from mmcv.visualization import color_val
import pickle
import os.path as osp
import os
import asyncio
import threading
from utils import get_logger

def img_detection(img, model,thre):
    result = inference_detector(model,img)
    hat_bbox,person_bbox = result[0],result[1]
    # 剔除分数过低的bbox
    hat_bbox,person_bbox = hat_bbox[hat_bbox[:,4] > thre],person_bbox[person_bbox[:,4]>thre]
    # 将阈值数据取出
    hat_bbox_pro = hat_bbox[:,4]
    hat_bbox = hat_bbox[:,0:4]
    person_bbox_pro = person_bbox[:,4]
    person_bbox = person_bbox[:,0:4]
    return (hat_bbox,hat_bbox_pro),(person_bbox,person_bbox_pro)

def imgs_detection(imgs, model,thre, step=1):
    """图片组识别
    
    Args:
        imgs ([type]): 图片列表
        thre ([int])：阈值
        model ([type]): 模型
        step (int, optional): 识别步长. Defaults to 1.
    
    Returns:
        [type]: [description]
    """    
    frames_index = [x for x in range(0,len(imgs),step)]
    hat_bboxs,person_bboxs = [] ,[]
    for img in imgs[frames_index]:
        result = inference_detector(model,img)
        hat_bbox,person_bbox = result[0],result[1]
        # 剔除分数过低的bbox
        hat_bbox,person_bbox = hat_bbox[hat_bbox[:,4] > thre],person_bbox[person_bbox[:,4]>thre]
        # 将阈值数据抹除
        hat_bbox = hat_bbox[:,0:4]
        hat_bboxs.append(hat_bbox)
        person_bbox = person_bbox[:,0:4]
        person_bboxs.append(person_bbox)
        
    hat_bboxs,person_bboxs = np.array(hat_bboxs),np.array(person_bboxs)
    return frames_index, hat_bboxs , person_bboxs


# @jit
# def track(bboxs, tracker):
#     """矩形框跟踪
    
#     Args:
#         bboxs ([type]): 矩形框数组
#         tracker ([type]): 跟踪器，默认为类中自带跟踪器
    
#     Returns:
#         [type]: 跟踪后的矩形框数组
#     """    
#     tracker_bboxs = [tracker.update(bbox) for bbox in bboxs]
#     return tracker_bboxs

def draw_label(imgs, bboxs, string, box_color, str_color):
        """为一组图像绘制标签
        
        Args:
            imgs ([type]): 待绘制图片组
            bboxs ([type]): bboxs：待绘制的矩形框数组
            bboxs_pro:每一个img中bboxs每个bbox的pro
            string ([type]): 待标记字符
            box_color ([type]): 矩形框颜色
            str_colore([type]): 字符串颜色
        
        Returns:
            [type]: 返回绘制好的图片组
        """    
        if len(bboxs) == 0 or len(bboxs) == 1:
            return imgs
        for i, img in enumerate(imgs):
            for r in bboxs[i]:
                cv2.rectangle(
                    img,
                    (int(r[0]),int(r[1])),
                    (int(r[2]),int(r[3])),
                    box_color,
                    2)
                cv2.putText(
                    img,
                    string + "|ID:" + str(int(r[4])),
                    (int(r[0]),int(r[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, str_color, 2) 
        return imgs
        
def draw_label1(imgs, bboxs,bboxs_pro, string, box_color, str_color):
        """为一组图像绘制标签
        
        Args:
            imgs ([type]): 待绘制图片组
            bboxs ([type]): bboxs：待绘制的矩形框数组
            bboxs_pro:每一个img中bboxs每个bbox的pro
            string ([type]): 待标记字符
            box_color ([type]): 矩形框颜色
            str_colore([type]): 字符串颜色
        
        Returns:
            [type]: 返回绘制好的图片组
        """    
        if len(bboxs[0]) == 0 or len(bboxs) == 1:
            return imgs
        bp = None
        if len(bboxs_pro) > 0:
            bp = bboxs_pro[0]
        for i, img in enumerate(imgs):
            for r in bboxs[i]:
                cv2.rectangle(
                    img,
                    (int(r[0]),int(r[1])),
                    (int(r[2]),int(r[3])),
                    box_color,
                    2)
                cv2.putText(
                    img,
                    string + "|ID:" + str(int(r[4]))+"|Pro:"+str(bp)[:4],
                    (int(r[0]),int(r[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, str_color, 2) 
        return imgs



def sifte(bboxs, size):
    """将出现次数极少的目标作为噪音剔除
    
    Args:
        bboxs ([type]): 需要去噪音的矩形框数组
        size ([type]): 筛孔大小，当某个目标在该bboxs数组中出现的次数少于size时，将会被剔除
    
    Returns:
        [type]: [description]
    """    
    tracker = Tracker()
    bbox_ids = track(bboxs, tracker)
    
    # 统计每个目标的寿命
    ids_num = {}
    for bbox_id in bbox_ids:
        for id in bbox_id[:,4]:
            if id not in ids_num.keys():
                ids_num[int(id)] = 0
            ids_num[int(id)] += 1
    # 筛选数据
    new_bboxs = []
    # 遍历所有帧
    for bbox_id in bbox_ids:
        new_bbox = []
        # 遍历每帧的i所有bbox框
        for i in bbox_id:
            if ids_num[int(i[4])] > size:
                new_bbox.append(i[0:4])
        new_bboxs.append(np.array(new_bbox))
    return np.array(new_bboxs)


def fill(bbox_ids, farmes=None):
    """
     填充未识别的帧
    Args:
        bbox_ids ([type]): 已经识别到的帧对应的bboxs框
        farmes ([type]): 已经识别到的帧对应的序号
    Returns:
        [type]: [description]
    """    
    new_bbox_ids = []
    new_farmes = []
    if len(bbox_ids) == 0:
        return new_bbox_ids, new_farmes
    farme_count = farmes[0]
    for i in range(len(bbox_ids)-1):
        # 将原来的第i帧放入新的bboxs框组中
        new_bbox_ids.append(bbox_ids[i])
        new_farmes.append(farme_count)
        farme_count += 1

        f_id = bbox_ids[i][:,4].astype(np.int32)    # 获得第i帧中的目标id
        b_id = bbox_ids[i + 1][:,4].astype(np.int32)# 获得第i+1帧中的目标id
        s_id = [x for x in f_id if x in b_id]       # 获得i与i+1帧共有的目标id

        # 获得第i帧和第i+1帧中共同的方框
        f_bbox = [x for x in bbox_ids[i] if int(x[-1]) in s_id]
        b_bbox = [x for x in bbox_ids[i+1] if int(x[-1]) in s_id]
        f_bbox = np.array(f_bbox).reshape(-1)   # 将方框压瘪为行向量以方便操作
        b_bbox = np.array(b_bbox).reshape(-1)

        d = farmes[i+1] - farmes[i]

        # 计算填充矩阵
        d_bboxs = np.zeros((d-1, len(f_bbox)))
        for i in range(len(f_bbox)):
            d_bboxs[:,i] = np.linspace(f_bbox[i], b_bbox[i], d, endpoint=False)[1:]
        
        # 矩阵填充
        for d_bbox in d_bboxs:
            new_bbox_ids.append(np.array(d_bbox.reshape(-1,5)))
            new_farmes.append(farme_count)
            farme_count += 1

    # 将最后一帧添加到新的bbox方框组：
    new_bbox_ids.append(bbox_ids[-1])
    new_farmes.append(farme_count)
    return new_farmes, new_bbox_ids


def over_step_video(video,step=1):
    """
    跳步视频
    
    Args:
        video ([VideoReader]): mmcv视频对象
        step (int, optional): 跳步大小. Defaults to 1.
    
    Yields:
        [frams]: [跳步片段]
    """    
    #assert isinstance(video,VideoReader)
    video_len = len(video)
    for start_index in range(0,video_len,step+1):
        end_index = start_index + step
        data = None
        if end_index >= video_len:
            data = video[start_index:]
        else:
            data = video[start_index:end_index]
        data = np.array(data)
        yield data


class DetectionSifter(object):
    def __init__(
        self,
        fps,
        video_name,
        resolution,
        connection,
        alive_thr=1,
        dead_thr=3,
        img_save_p='./static/pic',
        check_time=15):
        
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
        # self.img_save_p = osp.expanduser(img_save_p)
        self.img_save_p = img_save_p
        if not osp.exists(self.img_save_p):
            os.mkdir(self.img_save_p)
        self.check_thread = threading.Thread(target=self._check,args=(check_time,))
        self.check_thread.start()
        self.check_thread_break_point = False


    def add_object(self,bboxs,img,index):
        assert isinstance(bboxs,np.ndarray),'bbox is type is {0}'.format(type(bboxs))
        assert bboxs.shape[1] == 5
        self.process_step = index 
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
                if self._distance2center(bbox[0]) < self._distance2center(x['bbox'][bbi]):
                    # 刷新最佳bbox下标
                    x['best_bbox_index'] = len(x['bbox']) - 1
                    # 刷新保存的证据图像
                    x['img'] = self._draw_bbox(img,bbox)
                    # 刷新最佳图像保存时间
                    x['img_save_time'] = (self._get_time(),self.process_step / self.fps)
        
        # # 每隔三秒钟检测整个缓冲区，看是否需要保存或者清除一些object
        # if (self.process_step / self.fps) % 3 == 0:
        #     self._check_all_object()
    
    def _get_alive_time(self,detec_object):
        # 计算指定的object存活时间
        return len(detec_object['bbox']) / self.fps


    def _distance2center(self,bbox):
        # 计算指定的bbox到center的距离 
        x1,y1,x2,y2 = bbox
        bbox_center = (int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
        return math.sqrt(
            (self.center[0] - bbox_center[0])**2 + (self.center[1] - bbox_center[1])**2
        )

    def _check(self,check_time):
        while True:
            self._check_all_object()
            time.sleep(check_time)
            if self.check_thread_break_point:
                break
            
    def _check_all_object(self,is_last = False):
        if is_last:
            self.dead_thr = 0
        #Loger.info('检查缓冲区')
        # 检查缓存区的内容
        for id_num in self.buffer.copy():
            detec_object = self.buffer[id_num]
            # 获取探测到的目标的死亡时间(也就是跟踪的目标失去跟踪的时间)
            dead_time = self.process_step / self.fps - detec_object['over_time']
            # 如果死亡时间大于 self.dead_thr 秒,则可以认为该目标已经不会再出现在序列中
            if dead_time >= self.dead_thr:
                # 开始检查该目标所持续的时间(存活时间)
                live_time = self._get_alive_time(detec_object)
                #Loger.info('死亡时间大于３秒'+str(id_num))
                if live_time > self.alive_thr:
                    #Loger.info('{0}存活时间大于{1}秒'.format(str(id_num),self.alive_thr)+str(self.buffer.keys()))
                    #Loger.info('取出{0}放入数据库'.format(str(id_num)))
                    sp = osp.join(self.img_save_p,self.video_name+str(id_num)+'.jpg')
                    cv2.imwrite(sp,detec_object['img'])
                    doc = {
                            # 时间精确到秒
                            'time':detec_object['img_save_time'][0],
                            'info':{
                            # 图像二进制数据，需要用pick
                            'img':self.video_name+str(id_num)+'.jpg',
                            'img_shape':detec_object['img'].shape,
                            'video_name':self.video_name,
                            'video_time':detec_object['img_save_time'][1]
                            }
                    }
                    self.conn.insert_one(doc)
                    #Loger.info('Add a new record to db {0}'.format(doc['info']['img_shape']))
                    del self.buffer[id_num]
                else:
                    #Loger.info('{0}存活时间小于{1}秒'.format(str(id_num),self.alive_thr)+str(self.buffer.keys()))
                    #Loger.info('删除{0}'.format(str(id_num)))
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
        cv2.putText(
                    img,
                    'no wear helmet',
                    (int(bbox[0]),int(bbox[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8,
                    color_val('red'), 
                    2) 
        return img


    def _get_time(self):
        # 获取当前时间字符串
        ct  = time.localtime(time.time())
        current_time = '{0}-{1}-{2} {3}:{4}:{5}'.format(
            ct.tm_year,ct.tm_mon,ct.tm_mday,ct.tm_hour,ct.tm_min,ct.tm_sec)
        return current_time

    def clear(self):
        # 清理剩余的数据
        self.check_thread_break_point = True
        self._check_all_object(True)
