import cv2
import mmcv
from tqdm import tqdm
from cv2 import VideoWriter_fourcc
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='输入视频路径')
    parser.add_argument('--o',type=str,help='保存视频路径',default='./save.mp4')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.i):
        raise ValueError('输入视频不存在')
    elif os.path.basename(args.o).split('.')[1] != 'mp4':
        raise ValueError('保存视频格式必须为mp4')

    video = mmcv.VideoReader(args.i)
    resolution = video.resolution
    vwrite = cv2.VideoWriter(
        args.o,
        VideoWriter_fourcc(*'h264'),
        30,
        resolution)
    for frame in tqdm(video):
        vwrite.write(frame)
    print('processed over')

