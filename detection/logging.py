import datetime
from enum import Enum


class Color(Enum):
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class Level(Enum):
    DEBUG = '[DEBUG]'
    INFO = '[INFO]'
    SUCCESS = '[SUCCESS]'
    WARN= '[WARN]'
    ERROR = '[ERROR]'


class DetectionLogger:


    def __init__(self):
        self.__reset = "\033[0m"

    def log(self,color:Color,msg:str):
        """ Displays log messages the given message with given color."""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color.value} {current_time} -  {msg} {self.__reset}")

