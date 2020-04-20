from detection import inference_detector
import os
import cv2
from datetime import *
__all__=[inference_detector]

def testCaptureVideo():
    # !/usr/bin/python3


    ## opening videocapture
    cap = cv2.VideoCapture(0)

    ## some videowriter props
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(sz)

    fps = 30

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    ## open and set props
    vout = cv2.VideoWriter()
    vout.open('/home/dengjiafan/test/output.mp4', fourcc, fps, sz, True)

    cnt = 0
    startTime = datetime.now()
    while cnt < 120:
        cnt += 1
        print(cnt)
        _, frame = cap.read()
        if cnt==2:
            print(frame.shape)

        cv2.putText(frame, str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        vout.write(frame)
    endTime=datetime.now()
    print("花费时间：{0}".format((endTime-startTime).seconds))
    vout.release()
    cap.release()

def testCaptureImg():
    '''
    Opencv-python读取IP摄像头视频流/USB摄像头
    '''

    import cv2

    # 创建一个窗口 名字叫做Window
    cv2.namedWindow('Window', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    '''
    #打开USB摄像头
    cap = cv2.VideoCapture(0)
    '''

    # 摄像头的IP地址,http://用户名：密码@IP地址：端口/
    #ip_camera_url = 'http://admin:admin@192.168.1.101:8081/'
    # 创建一个VideoCapture
    cap = cv2.VideoCapture(0)

    print('IP摄像头是否开启： {}'.format(cap.isOpened()))

    # 显示缓存数
    print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    # 设置缓存区的大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 调节摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置FPS
    print('setfps', cap.set(cv2.CAP_PROP_FPS, 25))
    print("帧数",cap.get(cv2.CAP_PROP_FPS))
    idx=1
    startTime=datetime.now()
    while (True):
        # 逐帧捕获
        ret, frame = cap.read()  # 第一个参数返回一个布尔值（True/False），代表有没有读取到图片；第二个参数表示截取到一帧的图片
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Window', frame)
        if idx%30==0:
            endTime=datetime.now()
            detal=(endTime-startTime).seconds
            print("开始时间:{},结束时间:{}".format(startTime,endTime))
            startTime=endTime
            print("三十帧花费时间为{0}".format(detal))
        idx=(idx+1)%30
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 当一切结束后，释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()

def test():
    import cv2
    # 引入库
    cap = cv2.VideoCapture(0)
    idx = 0
    startTime = datetime.now()
    cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("帧数", cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, frame = cap.read()
        cv2.imshow("Video", frame)
        idx = (idx + 1) % 30
        if idx%30==0:
            endTime=datetime.now()
            detal=(endTime-startTime).seconds
            print("开始时间:{},结束时间:{}".format(startTime,endTime))
            startTime=endTime
            print("三十帧花费时间为{0}".format(detal))
        # 读取内容
        if cv2.waitKey(10) == ord("q"):
            break

    # 随时准备按q退出
    cap.release()
    cv2.destroyAllWindows()
    # 停止调用，关闭窗口

if __name__ == '__main__':
    test()