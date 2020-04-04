import numpy as np
from detection.mmdet.apis import inference_detector
import cv2
from . import Tracker,track
from numba import jit
from mmcv import VideoReader
import numpy as np


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
                    string + ":" + str(int(r[4])),
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


def fill(bbox_ids, farmes):
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
        return new_farmes, new_bbox_ids
    farme_count = 0
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