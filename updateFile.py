import pymongo
import os

# 更新数据库中图片命名数，方便下次查询   参数：ImageNumber的mongo连接器,图片命名数
def updateImageNum(collection,imageNum):
    collection.update_one({},{"$set":{"number":imageNum}})