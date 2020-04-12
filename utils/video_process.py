from mmcv import VideoReader,color_val
from tracker import Tracker,img_detection,track,fill,draw_label1,DetectionSifter
import numpy as np

class Process(object):
    def __init__(
        self,
        model,
        step=3,
        fourcc='avc1',
        thre=0.5):
        """视频处理对象

        Args:
            model (torch.Moudle): 目标探测模型
            step (int, optional): 跳帧探测步数,相当于加速(step-1)倍. Defaults to 3.
            fourcc (str, optional): 本地视频保存编码. Defaults to 'avc1'.
            thre (float, optional): 目标探测阈值. Defaults to 0.5.
        """        
      
        self.model = model
        self.step = step
        self.fourcc = fourcc
        self.thre = thre
        self.hat_color = 'green'
        self.person_color = 'red'
        self.psn_tracker = Tracker()
        self.hat_tracker = Tracker()
        self.last_frame_bbox = None # 用于保存每次调用模型前传获的bbox，避免下一次重复使用
        super().__init__()
    
    def _get_bboxs_from_imgs(self,imgs):
        """私有函数，用于获取每个imgs图像列表第一帧和最后一帧的bboxs
        
        Args:
            imgs (List):图像列表，每次传入的图像列表的最后一帧应该是下一个传入的图像列表的第一帧数
            
        
        Returns:
            (tuple,tuple): 第一帧和最后一帧的bboxs
        """        
        assert isinstance(imgs,(list,np.ndarray))
        assert len(imgs) == self.step,'输入图像列表的长度必须与step相等'
        first_frame_bboxs = None
        # 如果是第一次前传
        if self.last_frame_bbox is None:
            first_frame_bboxs = img_detection(imgs[0],self.model,self.thre)
            self.last_frame_bbox = img_detection(imgs[-1],self.model,self.thre)
        # 否则只需要前传最后一帧
        else:
            first_frame_bboxs = self.last_frame_bbox
            self.last_frame_bbox = img_detection(imgs[-1],self.model,self.thre)
        return first_frame_bboxs,self.last_frame_bbox

    def get_img(self,imgs,fl_indexs):
        """流式处理图像
        Example
        :imgs = video[0:3]
        :fl_indexs = [0,2]
        :imgs = Process().get_img(imgs,fl_indexs)
        Args:
            imgs ([List]): 图像数组，
            fl_indexs ([List]): 图像数组中每个图像对应视频中的第几帧
        
        Returns:
            [List]: 经过跟踪算法填充后的图像数组
        """        
        ffbbs,lfbbs = self._get_bboxs_from_imgs(imgs)
        (fhat_bbox,fhat_bbox_pro),(fperson_bbox,fperson_bbox_pro) = ffbbs
        (lhat_bbox,lhat_bbox_pro),(lperson_bbox,lperson_bbox_pro) = lfbbs
        flhat_bboxs = [fhat_bbox,lhat_bbox]
        flpsn_bboxs = [fperson_bbox,lperson_bbox]
        # 通过第一帧探测到的bboxs和最后一帧探测到的bboxs，结合跟踪算法推断出中间帧数的bboxs信息
        psn_bboxs_ids = track(flpsn_bboxs,self.psn_tracker)
        hat_bboxs_ids = track(flhat_bboxs,self.hat_tracker)
        #　跟踪推断中间的帧的bboxs
        psn_bboxs_index,all_psn_bbox_ids = fill(psn_bboxs_ids,fl_indexs)
        hat_bboxs_index,all_hat_bbox_ids = fill(hat_bboxs_ids,fl_indexs)
        # 将探测到的目标数据传入打包为objects
        psn_objects = [
            (all_psn_bbox_ids[bindex],all_psn_bbox_ids[bindex],psn_bboxs_index[bindex]) 
            for bindex in range(len(psn_bboxs_index)-1)]
        hat_objects = [
            (all_hat_bbox_ids[bindex],all_hat_bbox_ids[bindex],hat_bboxs_index[bindex]) 
            for bindex in range(len(hat_bboxs_index)-1)]
        # 绘制目标探测图像
        origin_frames = draw_label1(
            imgs,
            all_psn_bbox_ids,
            lperson_bbox_pro,
            "no wear helmet",
            color_val(self.person_color),
            color_val(self.person_color))
        origin_frames = draw_label1(
            imgs,
            all_hat_bbox_ids,
            lhat_bbox_pro,
            'wear helmet',
            color_val(self.hat_color),
            color_val(self.hat_color))
        
        return origin_frames,psn_objects,hat_objects

    


    



        
    
    

    