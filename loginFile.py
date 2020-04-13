import pymongo
import os


# 返回数据库集合,参数：数据库名，集合名
def Login(dataBaseName, collectionName):
    myclient = pymongo.MongoClient(host='127.0.0.1', port=27017)  # 指定主机和端口号创建客户端
    mydb = myclient[dataBaseName]  # 数据库使用
    mycol = mydb[collectionName]  # 表（集合）使用

    return mycol
