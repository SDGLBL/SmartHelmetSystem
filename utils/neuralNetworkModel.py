import argparse

from detection import init_detector

"""
该类用于获取神经网路模型，采用了单例模式，返回唯一的模型
"""
class NeuralNetworkModel(object):
    _haveInitialModel=False #模型是否被初始化
    _model=None             #模型对象
    _modelIsOccupied=False  #模型是否被占用
    """
    获取神经网络模型
    """
    def getModel(self):
        #判断模型是否被占用，如果被占用则返回None
        if self._modelIsOccupied:
            return None
        self._modelIsOccupied=True
        if self._haveInitialModel:
            return self._model
        else:
            self._haveInitialModel=True
            args = self._getParse_args()
            _model = init_detector(args.config, args.checkpoints, device='cuda:0')
            return _model
    """
    取消占用神经网络模型
    """
    def releaseModel(self):
        self._modelIsOccupied=False
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