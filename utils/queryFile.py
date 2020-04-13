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

# 返回全部信息  参数:Database的mongo连接器
def queryAllMessage(collection):
    data = collection.find({})
    return CursorTurnIntoList(data)

# 查询所有年月日
def queryAllTime(collection):
    datas = queryAllMessage(collection)
    dataBaseTime=set()
    for data in datas:
        yearMonthDay=data['time'].split(' ')[0]
        # print(yearMonthDay)
        HourMinuteSecond = data['time'].split(' ')[1]
        hour = HourMinuteSecond.split(':')[0]

        dataBaseTime.add(yearMonthDay + "-" + hour)

    dataBaseTime = list(dataBaseTime)
    dataBaseTime.sort()
    return dataBaseTime

# 根据list返回所有年月日   data为字典数据
def queryAllTimeAccordingList(datas):
    dataBaseTime = set()
    for data in datas:
        yearMonthDay=data['time'].split(' ')[0]
        HourMinuteSecond = data['time'].split(' ')[1]
        hour = HourMinuteSecond.split(':')[0]
        dataBaseTime.add(yearMonthDay+"-"+hour)

    dataBaseTime = list(dataBaseTime)
    dataBaseTime.sort()
    return dataBaseTime

# 根据单个约束条件,返回信息  参数：Database的mongo连接器，条件名，条件
def queryAboutCondition(collection, conditionName, condition):
    data = collection.find({conditionName:condition})
    return CursorTurnIntoList(data)

# 根据某些约束信息,查询数据   参数：Database的mongo连接器，约束条件
def queryAboutManyContidions(collection, contidions):
    expression = {}
    for key in contidions.keys():
        # 若约束条件为time，则为区间时间范围查找
        if str(key) == 'time':
            expression.setdefault(key,{'$gte': str(contidions[key][0]), '$lte': str(contidions[key][1])})
        # 若约束条件为图片名和视频名，则为精确查找
        elif str(key) == 'info.img' or str(key) == 'info.video_name':
            expression.setdefault(key,contidions[key])
        # 若约束条件为视频时间，则为区间范围查找
        elif str(key) == 'info.video_time':
            expression.setdefault(key,{'$gte': contidions[key][0], '$lte': contidions[key][1]})
        else: contidions

    data = collection.find(expression)
    return CursorTurnIntoList(data)

# 根据数据集返回图片的名称集  参数：数据库的数据集
def queryPictrues(dataSet):
    pic = []
    for data in dataSet:
        pic.append(data['info']['img'])
    return pic

# 根据年份查询  参数：Database的mongo连接器，年份
def queryAboutYear(collection,year):
    dataTimeStart = datetime(year, 1, 1, 0, 0, 0)
    dataTimeEnd = datetime(year, 12, 31, 23, 59, 59)
    expression = {'time':{'$gte':str(dataTimeStart),'$lte':str(dataTimeEnd)}}
    data = collection.find(expression)
    return CursorTurnIntoList(data)

# 根据月份查询  参数：Database的mongo连接器，年份，月份
def queryAboutMonth(collection,year,month):
    thisMonthDay = calendar.monthrange(year,month)[1]  # 算出这一月有多少天
    dataTimeStart = datetime(year,month,1,0,0,0)
    dataTimeEnd = datetime(year,month,thisMonthDay,23,59,59)
    expression = {'time':{'$gte':str(dataTimeStart),'$lte':str(dataTimeEnd)}}
    data = collection.find(expression)
    return CursorTurnIntoList(data)

# 根据日期查询  参数：Database的mongo连接器，年份，月份，日期
def queryAboutDay(collection,year,month,day):
    dataTimeStart = datetime(year, month, day, 0, 0, 0)
    dataTimeEnd = datetime(year, month, day, 23, 59, 59)

    expression = {'time': {'$gte': str(dataTimeStart), '$lte': str(dataTimeEnd)}}
    data = collection.find(expression)
    return CursorTurnIntoList(data)

# 根据小时查询  参数：Database的mongo连接器，年份，月份，日期，小时
def queryAboutHour(collection,year,month,day,hour):
    dataTimeStart = datetime(year, month, day, hour, 0, 0)
    dataTimeEnd = datetime(year, month, day, hour, 59, 59)

    expression = {'time': {'$gte': str(dataTimeStart), '$lte': str(dataTimeEnd)}}
    data = collection.find(expression)
    return CursorTurnIntoList(data)

# 返回图片命名递增到那个数字   # ImageNumber的mongo连接器，约束条件
def queryAboutImageNum(collection):
    return collection.find({})

# 将数据库数据转换成list，方便查询.
def CursorTurnIntoList(dataSet):
    new_data = []
    for data in dataSet:
        new_data.append(data)
    return new_data

