from app import getAllDiagramData
from utils.deleteFile import *
from utils.loginFile import *
from utils.videoHandler import*
from utils.videoHandler import *

imagesNum = 0 # 初始图片数量，为命名做准备

class VideoInfom(object):
    def __init__(self,index,title,videoURL,imgs):
        self.index=index
        self.title=title
        self.videoURL=videoURL
        self.imgs=imgs.__dict__

class Imgs(object):
    def __init__(self,idx,title,url):
        self.idx=idx
        self.title=title
        self.url=url

def getAllVideoName():
    import os
    for root, dirs, files in os.walk("static/video"):
        return files

if __name__ == '__main__':
    mycolImageNumber = Login('data','imageNumber')
    imagesNum = int(queryAboutImageNum(mycolImageNumber)[0]['number'])  # 获取初始名称
    mycolDatabase=Login('data','picAboutNotWearHat')
    deleteAllData(mycolDatabase)
    # videoHandler=VideoHandler()
    # videoHandler.addVideo("2020-04-03.mp4")
    # videoHandler.startHandleVideo()
    # allData=queryAllMessage(mycolDatabase)
    # print(allData)
    #timeList = queryAllTime(mycolDatabase)
    #print(timeList)
    #print(getAllDiagramData())
    #results = queryAboutDay(mycolDatabase,2020,3,16)
    # allTime=queryAllTime(mycolDatabase)
    # for time in allTime:
    #     print(time.split('-')[0])
   # print(type(allTime[0]))

    # print(queryPictrues(results))           # 获取查询后所有需要显示的图片
    # for result in results:
    #     print(result['time'])           #现实时间
    #     print(result['info']['img'])    #图片命名
    #     print(result['info']['video_name']) #图片视频名
    #     print(result['info']['video_time']) #图片在视频的时间
    #     print(result['info']['img_shape'])  # 图片形状
    #     print('='*20)

    # results=queryAllMessage(mycolDatabase)
    # print(results[2]['info']['video_name'])


