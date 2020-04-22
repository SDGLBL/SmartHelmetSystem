import subprocess as sp

"""
该类用于将图片写成rtmp流，发送到srs服务器上
"""
class RtmpStreamWriter(object):
    _pip=None
    """
    初始化管道
    """
    def __init__(self,width,height,fps,rtmpUrl):
        # ffmpeg command
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(width, height),
                   '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   rtmpUrl]
        # 管道配置
        self._pip = sp.Popen(command, stdin=sp.PIPE)


    """
    外部接口，写rtmp流
    """
    def write(self,frame):
        self._pip.stdin.write(frame.tostring())


    """
    外部接口，关闭管道
    """
    def close(self):
        self._pip.terminate()