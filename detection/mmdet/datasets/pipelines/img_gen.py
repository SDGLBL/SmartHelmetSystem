import numpy as np
from numpy import random
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms
from ..registry import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

# 随机通道变换
# Author:lijie

@PIPELINES.register_module
class RandomChangeChannel(object):
    """
    随机通道变换
    """
    def __init__(self):
        pass

    
    def _randomChangeChannel(self,img):

        if not isinstance(img, JpegImageFile):
            img = Image.fromarray(img)
        r, g, b = img.split()
        merge_list = [r, g, b]
        random.shuffle(merge_list)
        img = Image.merge('RGB', merge_list)
        img = np.array(img)
        return img

    def __call__(self, results):
        results['img'] = self._randomChangeChannel(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(RandomChangeChannel={})'.format(
            self.transformer.__repr__())


@PIPELINES.register_module
class RandomChangeBrightness(object):

    def __init__(self,brightness=0.5):
        """
        随机亮度变换
        :param brightness:How much to jitter brightness.
         brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
         or the given [min, max]. Should be non negative numbers.
        """
        self.transformer = transforms.Compose([
        transforms.ColorJitter(brightness=brightness)])

    def __call__(self, results):
        if not isinstance(results['img'],JpegImageFile):
            results['img'] = Image.fromarray(results['img'])
        results['img'] = self.transformer(results['img'])
        if not isinstance(results['img'],np.ndarray):
            results['img'] = np.array(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(RandomChangeBrightness={})'.format(
            self.transformer.__repr__())


@PIPELINES.register_module
class RandomChangeContrast(object):
    def __init__(self,contrast=0.2):
        """
        随机对比度变换
        :param contrast:How much to jitter contrast. contrast_factor is chosen uniformly
         from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
          Should be non negative numbers.
        """
        self.transformer = transforms.Compose([
        transforms.ColorJitter(contrast=contrast)])

    def __call__(self, results):
        if not isinstance(results['img'], JpegImageFile):
            results['img'] = Image.fromarray(results['img'])
        results['img'] = self.transformer(results['img'])
        if not isinstance(results['img'], np.ndarray):
            results['img'] = np.array(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(RandomChangeContrast={})'.format(
            self.transformer.__repr__())

@PIPELINES.register_module
class RandomChangeHue (object):
    def __init__(self,hue =0.2):
                
        """
        随机色调变换
        :param hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
         or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        """
        # 随机通道转换器材
        self.transformer = transforms.Compose([
        transforms.ColorJitter(hue =hue)])

    def __call__(self, results):
        if not isinstance(results['img'], JpegImageFile):
            results['img'] = Image.fromarray(results['img'])
        results['img'] = self.transformer(results['img'])
        if not isinstance(results['img'], np.ndarray):
            results['img'] = np.array(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(RandomChangeContrast={})'.format(
            self.transformer.__repr__())