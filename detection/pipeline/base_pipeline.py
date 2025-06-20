import cv2 as cv
import os
from detection.frame_parser import FrameParser
from detection.detectors import Detector
from detection.logging import DetectionLogger


class Pipeline:

    def __init__(self, videos: list, video_folder_path: str):
        """
        Initialize a pipeline.
        :param videos: List of video names with their extension to be included in the pipeline. e.g ["video1.mov", "video2.mov",...]
        :param video_folder_path: The path of the folder where the videos are located and be used during the pipeline.
        """
        self.video_folder_path = video_folder_path
        self.video_list = videos
        self.current_index = 0
        self.current_video = self.video_list[self.current_index]
        self.current_path = os.path.join(video_folder_path, self.current_video)
        self.detection_results = []

        self.logger = DetectionLogger()
        self.current_detector: Detector = None

    def get_frame_data(self) -> list[dict]:
        """
        Retrieves all frame data from the video as a list of dictionaries.

        The dictionary contains:

        - "timestamp" (str): The timestamp of the frame in the format 'HH:MM:SS',
          calculated based on the frame number and the video's FPS.

        - "frame" (numpy.ndarray): The actual video frame image (BGR format) as
          read by OpenCV.

        :return: List of dictionaries which are the frame data.
        """
        cap = cv.VideoCapture(self.current_path)
        parser = FrameParser(cap)
        frame_data = parser.parse()
        return frame_data

    def get_frames(self) -> list:
        """
        Retrieves the actual frame list from the frame data.
        :return: Frame list.
        """
        frames = [data['frame'] for data in self.get_frame_data()]
        return frames

    def next_video(self):
        """
        Updates the current index number and the current video.
        If the index is out of range, the index will be 0 again and also the video will be the first one.
        """
        self.current_index += 1
        if self.current_index >= len(self.video_list):
            self.current_index = 0
        self.current_video = self.video_list[self.current_index]
        self.current_path = os.path.join(self.video_folder_path, self.current_video)

    def start_detection(self):
        """
        Starts detection pipeline on the provided video list.
        """
        raise NotImplementedError()


    def log_detection_results(self):
        """
        Displays the results of the detection process.
        """
        raise NotImplementedError()

    def save_results(self,folder_path:str = 'results'):
        """
        Saves the detection results.
        """
        raise NotImplementedError()
