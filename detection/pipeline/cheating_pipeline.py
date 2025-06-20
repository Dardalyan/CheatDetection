import datetime
import json
import os
from detection import FaceDetector, ObjectDetector, SpeechDetector, TypingDetector
from detection.detectors.base_detector import Flag
from detection.logging import  Color, Level
from .base_pipeline import Pipeline


class CheatDetectionPipeline(Pipeline):

    def __init__(self, videos: list, video_folder_path: str):
        super().__init__(videos, video_folder_path)
        self.detection_results = [{'video': video, 'events': []} for video in self.video_list]

    def start_detection(self):
        """
        Starts detection pipeline on the provided video list.
        Detection process is a 4-layered process which are:
            1. face detection
            2. object detection
            3. speech detection and
            4. keyboard-typing detection.
        """
        count = 0

        while True:

            if count >= len(self.video_list):
                break

            self.logger.log(Color.MAGENTA, f" The Current Video is {self.current_video} ")
            self.logger.log(Color.MAGENTA,
                            f"--------------------------------------------------------------------------")

            frame_data = self.get_frame_data()

            # Face Detection -------------------------------------------------------------------------------------------
            self.logger.log(Color.CYAN,
                            f'{Level.INFO.value} Starting detection pipeline.. \n The 1st Layer : Face Detection')

            self.current_detector = FaceDetector(frame_data)
            self.current_detector.detect()
            self.detection_results[self.current_index]['events'].extend(self.current_detector.results)

            # Object Detection -----------------------------------------------------------------------------------------
            self.logger.log(Color.CYAN, f'{Level.INFO.value} The 2nd Layer: Object Detection')

            self.current_detector = ObjectDetector(frame_data)
            self.current_detector.detect()
            self.detection_results[self.current_index]['events'].extend(self.current_detector.results)

            # Speech Detection -----------------------------------------------------------------------------------------
            self.logger.log(Color.CYAN, f'{Level.INFO.value} The 3rd Layer: Speech Detection')

            self.current_detector = SpeechDetector(self.current_path,
                                                   self.current_video.split('.')[0] + '.wav',
                                                   'assets/audio')
            self.current_detector.detect()
            self.detection_results[self.current_index]['events'].extend(self.current_detector.results)

            # Typing Detection -----------------------------------------------------------------------------------------
            self.logger.log(Color.CYAN, f'{Level.INFO.value} The 4th Layer: Typing Detection')

            self.current_detector = TypingDetector(self.current_path,
                                                   self.current_video.split('.')[0] + '.wav',
                                                   'assets/audio')
            self.current_detector.detect()
            self.detection_results[self.current_index]['events'].extend(self.current_detector.results)
            # -----------------------------------------------------------------------------------------------------------

            # Sorting events by timestamp
            self.detection_results[self.current_index]['events'].sort(key=lambda r: datetime.datetime.strptime(r['timestamp'], '%H:%M:%S'))

            self.__add_summary(self.detection_results[self.current_index]['events'])

            count += 1
            self.next_video()

    def log_detection_results(self):

        print('\n')
        self.logger.log(Color.BLUE, f' -------------- RESULTS -------------- ')
        for result in self.detection_results:
            self.logger.log(Color.BLUE, f'VIDEO INFO: {result["video"]}')
            for r in result['events']:
                self.logger.log(Color.WHITE, r)
            self.logger.log(Color.MAGENTA, f'Summary: {result["summary"]}')
            self.logger.log(Color.BLUE, '--------------------------------------------------------')



    def save_results(self,folder_path:str = 'results'):

        os.makedirs(folder_path, exist_ok=True)

        for result in self.detection_results:
            file_name = result['video'].split('.')[0] + '.json'

            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)


    def __add_summary(self, events: list[dict]):
        """
        Adds the summary of the detection results for each video.
        :param events: The list of dictionaries which contain 'timestamp' and 'event_type'.
        """

        summary = []

        if sum(e['event_type'] == Flag.LOOKING_AWAY.value for e in events) >= 5:
            summary.append("There were several moments when the subject looked away from the screen.")
        elif sum(e['event_type'] == Flag.LOOKING_AWAY.value for e in events) > 0:
            summary.append("The subject briefly looked away from the screen a few times.")

        if sum(e['event_type'] == Flag.MULTIPLE_PEOPLE.value for e in events) >= 2:
            summary.append("Multiple individuals were detected in the frame on several occasions.")
        elif sum(e['event_type'] == Flag.MULTIPLE_PEOPLE.value for e in events) == 1:
            summary.append("Another person appeared in the frame at one point.")

        if sum(e['event_type'] == Flag.NO_PEOPLE.value for e in events) >= 5:
            summary.append("The subject was missing from the camera view for an extended period.")
        elif sum(e['event_type'] == Flag.NO_PEOPLE.value for e in events) > 0:
            summary.append("The subject was temporarily not visible in the video.")

        if sum(e['event_type'] == Flag.EXTERNAL_DEVICE_DETECTED.value for e in events) >= 2:
            summary.append("An unauthorized device was visible in the video multiple times.")
        elif sum(e['event_type'] == Flag.EXTERNAL_DEVICE_DETECTED.value for e in events) == 1:
            summary.append("An external device was detected at one point during the session.")

        if sum(e['event_type'] == Flag.KEYBOARD_TYPING_DETECTED.value for e in events) >= 5:
            summary.append(
                "Frequent keyboard typing sounds were detected, suggesting possible unauthorized computer use.")
        elif sum(e['event_type'] == Flag.KEYBOARD_TYPING_DETECTED.value for e in events) > 0:
            summary.append("Some keyboard typing sounds were detected during the session.")

        if sum(e['event_type'] == Flag.SPEECH_DETECTED.value for e in events) >= 5:
            summary.append("Extended periods of speech were detected, possibly indicating conversation.")
        elif sum(e['event_type'] == Flag.SPEECH_DETECTED.value for e in events) > 0:
            summary.append("Occasional speech was detected, which may indicate communication.")

        if not summary:
            summary.append(
                "No suspicious activity was detected during the session. "
                "The subject appeared to remain attentive and compliant throughout the video.")

        self.detection_results[self.current_index]['summary'] = " ".join(summary)





