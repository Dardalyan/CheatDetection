import cv2 as cv
import datetime
from detection.logging import DetectionLogger, Color, Level


class FrameParser:

    def __init__(self, capture:cv.VideoCapture):
        self.cap = capture
        self.fps = capture.get(cv.CAP_PROP_FPS)
        self.frames:list[dict] = []
        self.total_time = 0
        self.total_frame = 0

    def parse(self) -> list:
        """
        Parses the given capture object (VideoCapture).
        The time period is arranged as 1 second.
        :return: The list of frames which contains 'timestamp' (str) and 'frame' (numpy.ndarray)
        """
        logger = DetectionLogger()
        success, frame = self.cap.read()
        count = 0

        while success:

            if count % int(self.fps) == 0:
                current_sec = count / int(self.fps)
                timestamp = str(datetime.timedelta(seconds=current_sec))

                data = {
                    "timestamp": timestamp,
                    "frame": frame,
                }
                self.frames.append(data)

            success, frame = self.cap.read()
            count += 1


        self.cap.release()
        self.total_frame = count
        self.total_time = self.total_frame / self.fps

        logger.log(Color.GREEN, f"{Level.SUCCESS.value} Frame Parsing Completed ")

        return self.frames

