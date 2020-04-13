import pymongo
import json
from PIL import Image
from io import BytesIO,StringIO
import pickle
from bson.json_util import dumps,loads
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import calendar
from queryFile import *
from updateFile import *


def deleteVideo(collection, videoName):
    videoDir="video/"           # 默认存储路径
    print(videoDir+(videoName))
    videoUrl=videoDir+(videoName)
    if os.path.exists(videoUrl):
        deleteAboutVideoName(collection,videoName)
        os.remove(videoUrl)

# 删除数据根据任意条件   参数:Database的mongo连接器,约束条件
def deleteAboutManyContidions(collection, contidions):
    # 删除本地图片
    deleteData = queryAboutManyContidions(collection,contidions)
    pic = queryPictrues(deleteData)
    deleteLocalPic(pic)     # 删除本地图片

    # 删除数据库信息
    expression = {}
    for key in contidions.keys():
        # 若约束条件为time，则为区间时间范围查找
        if str(key) == 'time':
            expression.setdefault(key, {'$gte': str(contidions[key][0]), '$lte': str(contidions[key][1])})
        # 若约束条件为图片名或视频名，则为精确查找
        elif str(key) == 'info.img' or str(key) == 'info.video_name':
            expression.setdefault(key, contidions[key])
        # 若约束条件为视频时间，则为区间范围查找
        elif str(key) == 'info.video_time':
            expression.setdefault(key, {'$gte': contidions[key][0], '$lte': contidions[key][1]})
        else:
            contidions
    collection.delete_many(expression)

# 删除所有数据  参数:Database的mongo连接器
def deleteAllData(collection):
    # 删除本地所有图片
    deleteData = queryAllMessage(collection)
    pic = queryPictrues(deleteData)
    deleteLocalPic(pic)

    collection.delete_many({})

# 删除数据，根据视频源  参数:Database的mongo连接器，音频名字
def deleteAboutVideoName(collection,video_name):
    # 删除本地图片
    deleteData = queryAboutCondition(collection,'info.video_name',video_name)
    pic = queryPictrues(deleteData)
    deleteLocalPic(pic)

    # 删除数据库信息
    expression ={'info.video_name':video_name}
    collection.delete_many(expression)


# 根据年份删除  参数：Database的mongo连接器，年份
def deleteAboutYear(collection,year):
    dataTimeStart = datetime(year, 1, 1, 0, 0, 0)
    dataTimeEnd = datetime(year, 12, 31, 23, 59, 59)
    expression = {'time':{'$gte':str(dataTimeStart),'$lte':str(dataTimeEnd)}}
    data = collection.delete_many(expression)
    return data

# 根据月份删除  参数：Database的mongo连接器，年份，月份
def deleteAboutMonth(collection,year,month):
    thisMonthDay = calendar.monthrange(year,month)[1]  # 算出这一月有多少天
    dataTimeStart = datetime(year,month,1,0,0,0)
    dataTimeEnd = datetime(year,month,thisMonthDay,23,59,59)
    expression = {'time':{'$gte':str(dataTimeStart),'$lte':str(dataTimeEnd)}}
    data = collection.delete_many(expression)
    return data

# 根据日期删除  参数：Database的mongo连接器，年份，月份，日期
def deleteAboutDay(collection,year,month,day):
    dataTimeStart = datetime(year, month, day, 0, 0, 0)
    dataTimeEnd = datetime(year, month, day, 23, 59, 59)
    expression = {'time': {'$gte': str(dataTimeStart), '$lte': str(dataTimeEnd)}}
    data = collection.delete_many(expression)
    return data


# 删除文件夹中的图片   参数:删除的图片名字集合
def deleteLocalPic(deletePics):
    for deletePic in deletePics:
        path = './static/pic/'+deletePic
        print(path,'='*10,"delete")
        if not os.path.exists(path): continue
        os.remove(path)
