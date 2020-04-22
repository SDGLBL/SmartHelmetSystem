import argparse

from detection import init_detector

"""
该类用于获取神经网路模型，采用了单例模式，返回唯一的模型
"""
class NeuralNetworkModelManager(object):
    _haveInitialModel=False #模型是否被初始化
    _model=None             #模型对象
    _isOccupied=False  #模型是否被占用
    _password=None          #占用模型和释放模型使用的密码
    _flag=True
    _instant=None
    def __new__(cls, *args, **kwargs):
        if cls._instant is None:
            cls._instant = super().__new__(cls)
        return cls._instant
    def __init__(self):
        if not NeuralNetworkModelManager._flag:
            return
        NeuralNetworkModelManager._flag = False
    """
    获取神经网络模型
    """
    def getModel(self,password):
        #判断模型密码释放正确,模型是否被占用
        if password is not self._password or not self._isOccupied :
            return None
        self._isOccupied=True
        if self._haveInitialModel:
            return self._model
        else:
            self._haveInitialModel=True
            args = self._getParse_args()
            _model = init_detector(args.config, args.checkpoints, device='cuda:0')
            return _model

    """
    占用神经网络,如果成功则返回True，否则返回False
    """
    def occupyModel(self,password):
        if self._isOccupied:
            return False
        else:
            self._password=password
            self._isOccupied=True
            return True
    """
    取消占用神经网络模型,成功则返回True，否则返回False
    """
    def releaseModel(self,password):
        if self._password==password:
            self._isOccupied = False
            return True
        else:
            return False

    """
    获取神经网络模型的参数
    """
    def _getParse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='./detection/myconfigs/baseline/faster_rcnn_r50_fpn_1x.py')
        parser.add_argument('--checkpoints', default='./fastr50_963.pth')
        parser.add_argument('--i', type=str, help='视频文件路径', default='./test.mp4')
        parser.add_argument('--thre', type=float, help='目标阈值', default=0.5)
        parser.add_argument('--o', type=str, help='视频文件输出路径', default='./test_out.mp4')
        parser.add_argument('--fps', type=int, default=30, help='输出的视频fps')
        parser.add_argument('--hat_color', type=str, default='green', help='安全帽框颜色')
        parser.add_argument('--person_color', type=str, default='red', help='人头框颜色')
        args = parser.parse_args()
        return args