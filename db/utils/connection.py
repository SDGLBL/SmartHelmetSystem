import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import motor
from pymongo import MongoClient
import time

def get_collection(asyn=False,dbname='safehat',time=time.localtime(time.time())):
        
    """
    获取mongo数据库连接(异步连接)
    
    Args:
        asyn (bool, optional): 是否为异步连接. Defaults to False.
        dbname (str, optional): 数据库名称. Defaults to 'safehat'.
        time ([type], optional): 当前系统时间. Defaults to time.localtime(time.time()).
    
    Returns:
        [type]: [数据库connection]
    """    
    client = None
    # 函数每次调用都会自动查询当前时间
    current_time = str(time.tm_year)+'-'+str(time.tm_mon)+'-'+str(time.tm_mday)
    if not asyn:
        client = MongoClient()
        db = client[dbname]
        if current_time not in db.list_collection_names():
            db.create_collection(current_time)
        collection = db[current_time]
        return collection
    else:
        client = AsyncIOMotorClient('mongodb://localhost:27017')
        db = client[dbname]
        collection = db[current_time]
        return collection