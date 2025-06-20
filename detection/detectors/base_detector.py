import urllib
from enum import Enum
import shutil
from moviepy.audio.io.AudioFileClip import AudioFileClip
import os
import tensorflow_hub as hub
import csv
from detection.logging import DetectionLogger, Color, Level


class Flag(Enum):

    LOOKING_AWAY = 'LOOKING_AWAY'
    MULTIPLE_PEOPLE = 'MULTIPLE_PEOPLE'
    NO_PEOPLE = 'NO_PEOPLE'
    EXTERNAL_DEVICE_DETECTED = 'EXTERNAL_DEVICE_DETECTED'
    KEYBOARD_TYPING_DETECTED = 'KEYBOARD_TYPING_DETECTED'
    SPEECH_DETECTED = 'SPEECH_DETECTED'

class Detector:

    def __init__(self):
        self.results = []
        self.logger =  DetectionLogger()

    def detect(self):
        raise NotImplementedError()


class VideoDetector(Detector):

    def __init__(self, frames: list[dict]):
        """
        Initialize an VideoDetector.

        :param frames: List of dictionaries that keeps frame data. It is highly recommended to FrameParser's parse() method.
        This is because list of dictionaries must contain 'timestamp' (str) and 'frame' (numpy.ndarray).
        """
        super().__init__()
        self.frames = frames

    def detect(self):
        raise NotImplementedError()

class AudioDetector(Detector):

    def __init__(self, video_path: str, target_file_name: str, target_folder:str = "audio"):
        """
        Initialize an AudioDetector.

        Extracts audio frames from a video file and stores the file in target folder name which is "audio" as a default.

        :param video_path: The exact video file path, e.g. "assets/video/example_video1.mov" or "assets/video/example_video1.mp4"
        :param target_file_name: The target audio file name, e.g. "example_audio1.wav"
        :param target_folder: The target audio folder name, e.g. "audio" or "assets/audio"
        """
        super().__init__()
        self.target_folder = target_folder
        self.audio_path = os.path.join(target_folder, target_file_name)

        audio_clip = AudioFileClip(video_path)

        if self.__file_exists():
            self.logger.log(Color.RED, f'{Level.WARN.value} The file {self.audio_path} already existed and was overwritten!')

        audio_clip.write_audiofile(self.audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
        audio_clip.close()

        self.model = None
        self.labels = self.__make_model_ready()
        self.frame_duration = 1.0

    def __create_folder_if_missing(self):
        """
        Creates folder with a provided folder name if it doesn't exist.
        """

        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        else:
            if not os.path.isdir(self.target_folder):
                shutil.rmtree(self.target_folder)
                os.makedirs(self.target_folder)

    def __file_exists(self) -> bool:
        """
        Checks whether the file exists or not.
        """
        self.__create_folder_if_missing()

        if os.path.exists(self.audio_path):
            return True
        else:
            return False

    def __make_model_ready(self) -> list:
        """
        Downloads and loads 'YAMNET' model via the public link.
        Fetches the labels for the model and stores them in 'yamnet_class_map.csv' file
        that will be located in the root directory.

        :return: List of labels.
        """

        yamnet_model = 'https://tfhub.dev/google/yamnet/1'
        self.model = hub.load(yamnet_model)

        labels_path = 'yamnet_class_map.csv'
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
            labels_path
        )

        with open(labels_path) as f:
            reader = csv.reader(f)
            next(reader)
            labels = [row[2] for row in reader]

        return labels

    def detect(self):
        raise NotImplementedError()

