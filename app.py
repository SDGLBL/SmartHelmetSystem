from flask import Flask, render_template,request
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField
from  wtforms import validators
from flask import jsonify
from flask_cors import *
import json
from datetime import *
from loginFile import *
from queryFile import *
from queryFile import *
from insertFile import *

app = Flask(__name__,static_url_path="")
CORS(app, supports_credentials=True)
app.secret_key='djf'

#资源访
sourcesHost= 'http://data.webbanc.xyz:8080'
#sourcesHost='http://127.0.0.1:5000'
#主页面



@app.route('/',methods=['GET'])
def hello_world():
    return render_template('mainPage.html')

@app.route('/login')
def toLogin():
    return render_template('login.html')


@app.route('/testPost',methods=['POST'])
def TestPost():
    if request.method=='POST':
        name=request.values.get("name")
        print(name)
        return jsonify({"url":"http://192.168.3.28:5000/tiger.jpg"})

#返回视频数据
@app.route('/grabVideoData',methods=['GET','POST'])
def getVideoData():
    print("连接成功,准备回传视频数据"+request.method)
    videoNames=getAllVideoName()
    print(videoNames)
    videoInfos=[]
    #连接数据库
    mycolImageNumber = Login('data', 'imageNumber')
    imagesNum = int(queryAboutImageNum(mycolImageNumber)[0]['number'])  # 获取初始名称
    mycolDatabase = Login('data', 'picAboutNotWearHat')
    idx_video = 0
    for videoName in videoNames:
        videoName=videoName.split('.')[0]
        #根据视频名字获取图片
        results=queryAboutCondition(mycolDatabase,'info.video_name',videoName)
        imgInfos=[]
        idx_img = 0
        for result in results:
            #创建图片信息
            img=ImgInfo(idx_img,result['time'],getImgUrl(result['info']['img']))
            #将图片信息加入的列表中
            imgInfos.append(img.__dict__)
            # imgInfos.append({"idx":idx_img,
            #                  "title":result['time'],
            #                  "url":getImgUrl(result['info']['img'])})
            idx_img=idx_img+1
        #创建视频信息
        videoInfom=VideoInfom(idx_video,videoName,getVideoUrl(videoName),imgInfos)
        #将视频信息加入列表
        videoInfos.append(videoInfom.__dict__)
        # videoInfos.append({"index":idx_video,
        #                    "tilte":videoName,
        #                    "videoURL":getVideoUrl(videoName),
        #                    "imgs":imgInfos})
        idx_video=idx_video+1
    print("视频数据"+json.dumps(videoInfos))
    return  json.dumps(videoInfos)
#返回统计表格数据
@app.route('/grabDiagramData',methods=['GET','POST'])
def getDiagramData():
    print("连接成功，准备回传统计数据")
    #连接数据库
    mycolImageNumber = Login('data', 'imageNumber')
    imagesNum = int(queryAboutImageNum(mycolImageNumber)[0]['number'])  # 获取初始名称
    mycolDatabase = Login('data', 'picAboutNotWearHat')
    idx=0
    diagramDataList=[]
    #获取所有日期
    allTimes=queryAllTime(mycolDatabase)
    for time in allTimes:
        timeStrs=time.split('-')
        results=queryAboutDay(mycolDatabase,int(timeStrs[0]),int(timeStrs[1]),int(timeStrs[2]))
        #创建统计数据
        diagramDataList.append({"index":idx,
                                "date":time,
                                "totalBreak":len(results)})
        idx=idx+1
    diagramDataList.append({"index":1,
                                "date":"2020-04-02",
                                "totalBreak":40})
    print("统计数据："+json.dumps(diagramDataList))
    return json.dumps(diagramDataList)


#返回所有视频的名字
def getAllVideoName():
    import os
    for root, dirs, files in os.walk("./static/video"):
        return files
#根据视频名字返回视频路径
def getVideoUrl(videoName):
    #return 'http://'+serverIP+':'+serverPort+"/video/"+videoName
    return sourcesHost + "/video/" + videoName+".mp4"
#根据图片名字返回图片路径
def getImgUrl(imgName):
    #return 'http://'+serverIP+':'+serverPort+"/pic/"+imgName
    return sourcesHost + "/pic/" + imgName

#获取统计数据
def getAllDiagramData():
    # 连接数据库
    mycolImageNumber = Login('data', 'imageNumber')
    imagesNum = int(queryAboutImageNum(mycolImageNumber)[0]['number'])  # 获取初始名称
    mycolDatabase = Login('data', 'picAboutNotWearHat')
    #获取起始和末尾的时间
    timeList=queryAllTime(mycolDatabase)
    startTime=timeList[0]
    endTime=timeList[len(timeList)-1]

    startTimeStrs=startTime.split("-")
    endTimeStrs=endTime.split("-")

    startDate=datetime(int(startTimeStrs[0]),int(startTimeStrs[1]),int(startTimeStrs[2]))
    endDate=datetime(int(endTimeStrs[0]),int(endTimeStrs[1]),int(endTimeStrs[2]))
    #endDate=datetime(2020,4,2)
    currentDate=datetime(int(startTimeStrs[0]),int(startTimeStrs[1]),int(startTimeStrs[2]))

    diagramDatas_day=[]
    idx_day=0
    while(currentDate<=endDate):
        result_days=queryAboutDay(mycolDatabase,currentDate.year,currentDate.month,currentDate.day)
        #一天违规的次数
        totalBreak_day=-1;
        if(len(result_days)!=0):
            totalBreak_day=len(result_days)

        #一天中24小时的违规记录
        diagramDatas_hour = []
        for idx_hour in range(0,24):
            result_hours=queryAboutHour(mycolDatabase,currentDate.year,currentDate.month,currentDate.day,idx_hour)
            totalbreak_hour=-1;
            if(len(result_hours)!=0):
                totalbreak_hour=len(result_hours)
            diagramDatas_hour.append({"idx":idx_hour,
                                      "totalBreak":totalbreak_hour})
        diagramDatas_day.append({"index":idx_day,
                                 "date":currentDate.strftime('%Y-%m-%d'),
                                 "totalBreak":totalBreak_day,
                                 "hours":diagramDatas_hour})
        idx_day=idx_day+1
        currentDate=currentDate+timedelta(days=1)
    return diagramDatas_day

#存放视频信息，包括视频名字、路径、所包含的图片的信息
class VideoInfom(object):
    def __init__(self,index,title,videoURL,imgs):
        self.index=index
        self.title=title
        self.videoURL=videoURL
        self.imgs=imgs
#存放图片信息
class ImgInfo(object):
    def __init__(self,idx,title,url):
        self.idx=idx
        self.title=title
        self.url=url


#存放统计信息
class DiagramData(object):
    def __init__(self,index,date,totalBreak):
        self.index=index
        self.data=date
        self.totalBreak=totalBreak


if __name__ == '__main__':


    print(getAllDiagramData())
    app.run()
    jsonify([{"index":0,
              "date":"2020-04-01",
              "totalBreak":28,
              "hours":[{"idx":"1",
                        "totalBreak":15},
                       {"idx":"2",
                        "totalBreak":-1},
                       {"idx":"3",
                        "totalBreak":13}]},
             {"index": 1,
              "date": "2020-04-02",
              "totalBreak":-1,
              "hours": [{"idx": "1",
                         "totalBreak": -1},
                        {"idx": "2",
                         "totalBreak": -1},
                        {"idx": "3",
                         "totalBreak": -1}]},
             {"index": 2,
              "date": "2020-04-03",
              "totalBreak":48,
              "hours": [{"idx": "1",
                         "totalBreak": 15},
                        {"idx": "2",
                         "totalBreak": 16},
                        {"idx": "3",
                         "totalBreak": 17}]}])
