import json
import pickle
from bson.json_util import loads
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
from random import randint
from tools.updateFile import *

# 根据本地文件,插入其中的json数据   参数：Database的mongo连接器，ImageNumber的mongo连接器，本地json的地址，图片命名数
def insertAboutLocalJson(collectionDataBase,collectionImageNum, jsonUrl,imagesNum):
    if not os.path.exists(jsonUrl):
        print("不存在此json文件!")
        return imagesNum

    items=[]
    plt.axis('off')
    # 读取json文件
    with open(jsonUrl, 'r', encoding='utf-8')as f:
        for item in f.readlines():
            items.append(json.loads(item))

    # 分析数据
    for i,item in enumerate(items):
        print(i,"正在输入,共",len(items),"个")
        data = loads(json.dumps(item))

        # 更改数据 img
        ByteImage = data['info']['img']
        ArrayImage = pickle.loads(ByteImage)

        plt.imshow(ArrayImage)
        plt.savefig('pic/%d.jpg' % (imagesNum))
        data['info']['img']=str("%d.jpg" % imagesNum)
        imagesNum += 1                                          # 命名数更改

        # 更改数据 time
        data['time'] = changeTimeFormat(data['time'])           # 更改时间格式

        data.pop('_id', None)  # 删除id，为转化json做准备
        # print(data)
        insertOneData(collectionDataBase, data)                 # 插入数据
        imagesNum = insertRamdomData(collectionDataBase,data,imagesNum)



    updateImageNum(collectionImageNum, imagesNum)               # 更新数量
    # 更新图片命名数
    return imagesNum

# 插入数据
def insertData(collection, newJsonData):
    collection.insert_many(newJsonData)

def insertOneData(collection,newJsonData):
    collection.insert_one(newJsonData)

# 因为输入格式有问题，例如3应该是03才能正确比对字符串大小
def changeTimeFormat(jsonDateTime):

    jsonDateSplit = jsonDateTime.split(' ')
    yearMonthDay = jsonDateSplit[0].split('-')
    hourMinuteSecond = jsonDateSplit[1].split(':')

    yearMonthDay = [int(item) for item in yearMonthDay]
    hourMinuteSecond = [int(item) for item in hourMinuteSecond]

    standardTimeFormat = \
        datetime(yearMonthDay[0],yearMonthDay[1],yearMonthDay[2],
                 hourMinuteSecond[0],hourMinuteSecond[1],hourMinuteSecond[2])
    return str(standardTimeFormat)

def randomTime(year=None,month=None,day=None):
    if year == None: year = randint(2019,2020)  # 年份限制于2019-2020
    if month == None: month=randint(1,12) if year == 2019 else randint(1,3)
    thisMonthDay = calendar.monthrange(year, month)[1]  # 根据这个月的天数
    if day ==None: day =randint(1,thisMonthDay)

    hour = randint(0,23)
    minute = randint(0,59)
    second = randint(0,59)

    randomDateTime = datetime(year,month,day,hour,minute,second)
    return str(randomDateTime)

def insertRamdomData(collection, data, imagesNum,times=24):
    if '_id' in data: data.pop('_id')
    for i in range(times):
        data['time'] = randomTime()     # 随机时间
        data['info']['img'] = str("%d.jpg" % imagesNum)
        plt.savefig('pic/%d.jpg' % (imagesNum))     # 再次保存图片
        imagesNum += 1

        insertOneData(collection, data)  # 插入数据
    return imagesNum

