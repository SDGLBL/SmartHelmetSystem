import logging

def get_logger(
    filename='log.txt'
    ,filemode="w"
    , format="%(asctime)s %(name)s:%(levelname)s:%(message)s"
    , datefmt="%d-%M-%Y %H:%M:%S"):

    logging.basicConfig(level=logging.INFO,filename='log.txt',filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")
    return logging.getLogger(__name__)