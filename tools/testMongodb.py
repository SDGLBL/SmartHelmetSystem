from utils.deleteFile import *
from utils.loginFile import *

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
    for root, dirs, files in os.walk("../static/video"):
        return files

if __name__ == '__main__':
    mycolImageNumber = Login('data','imageNumber')
    imagesNum = int(queryAboutImageNum(mycolImageNumber)[0]['number'])  # 获取初始名称
    mycolDatabase=Login('data','picAboutNotWearHat')

    # now = datetime.datetime(2017,6,22,20,23)
    # date = now + datetime.timedelta(hours=1)
    # print(date.strftime('%Y-%m-%d-%H'))
    # print(date)
    #
    # d1=datetime.datetime(2017,6,6,3)
    # d2=datetime.datetime(2017,6,6)
    # print(d1>d2)
    for i in range(1,25):
        print(i)

    # result=queryAboutDay(mycolDatabase,2020,4,2)
    # print(len(result))
    #results = queryAboutCondition(mycolDatabase,'info.video_name','test_out.mp4')
    #results =queryAllMessage(mycolDatabase)
    #print(len(results))
    #print(results[0]['info']['video_name'])
    # for result in results:
    #     i=0
    #     imgs=Imgs()
    #     i=i+1


    #deleteAllData(mycolDatabase)
    #imagesNum = insertAboutLocalJson(mycolDatabase,mycolImageNumber,'./data/data.json',imagesNum)

    #deleteAboutManyContidions(mycolDatabase,{'time':[datetime(2020,3,16,21,14),datetime(2020,3,16,21,15,17)],'info.video_time':[30.0,35.0]})

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


