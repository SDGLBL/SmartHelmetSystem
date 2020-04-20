from flask import Flask, render_template,request
from flask import jsonify
from flask_cors import *
from datetime import *
import time
import queue
from utils import get_logger
from utils.loginFile import *
from utils.queryFile import *
from utils.insertFile import *
from flask_socketio import SocketIO
from geventwebsocket.handler import WebSocketHandler         #提供WS（websocket）协议处理
from geventwebsocket.server import WSGIServer                #websocket服务承载
#WSGIServer导入的就是gevent.pywsgi中的类
from gevent.pywsgi import WSGIServer
from flask import Flask, request,make_response,render_template, redirect, url_for
from werkzeug.utils import secure_filename
from os import path
from geventwebsocket.websocket import WebSocket
import _thread

app = Flask(__name__,static_url_path="")
#CORS(app, supports_credentials=True)
app.secret_key='djf'
socketio = SocketIO(app)

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
#上传文件
@app.route("/uploadVideos",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files["file"]
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path,'static/uploads/')
        file_name = upload_path + secure_filename(f.filename)
        f.save(file_name)
        return redirect(url_for('upload'))
    else:
        return "失败"
#连接客户端的socket
@app.route('/connect',methods=['GET','POST'])
def connectSocket():
    client_socket = request.environ.get('wsgi.websocket')
    if not client_socket:
        print("没有连接socket")
    else:
        print("连接socket成功")
    #time.sleep(5)
    print("准备通知前端更新")
    client_socket.send(json.dumps("连接成功"))
    return "ok"


#返回视频、截图数据
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

            idx_img=idx_img+1
        #创建视频信息
        videoInfom=VideoInfom(idx_video,videoName,getVideoUrl(videoName),imgInfos)
        #将视频信息加入列表
        videoInfos.append(videoInfom.__dict__)

        idx_video=idx_video+1
    print("视频数据"+json.dumps(videoInfos))
    return  json.dumps(videoInfos)
#返回统计表格数据
@app.route('/grabDiagramData',methods=['GET','POST'])
def getDiagramData():

    allDiagramData=getAllDiagramData()
    print(allDiagramData)
    return json.dumps(getAllDiagramData())



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

    diagramDatas_day=[]
    idx_day=0
    for timeStr in timeList:
        timeStrs=timeStr.split("-")
        currentDate = datetime(int(timeStrs[0]), int(timeStrs[1]), int(timeStrs[2]))
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
    # _thread.start_new_thread(hello, (), {})
    Loger = get_logger()
    Img_Q = queue.Queue(100)
    http_server = WSGIServer(('127.0.0.1', 5000), application=app, handler_class=WebSocketHandler)
    http_server.serve_forever()
