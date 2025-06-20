from ultralytics import YOLO
from .base_detector import Flag, VideoDetector
from detection.logging import  Color, Level


class ObjectDetector(VideoDetector):

    def __init__(self, frames: list[dict]):
        super().__init__(frames)

        self.model_m = YOLO('../../yolov8m.pt')
        self.model_s = YOLO('../../yolov8n.pt')

        self.target_objects = {'cell phone', 'keyboard', 'book', 'laptop', 'pen', 'tablet'}

    def detect(self):
        """
        Detects objects in provided frames.
        Provided frame data list should be parsed via FrameParser.
        It is highly recommended because the expected frame list must be the list of dictionaries which
        contains 'timestamp' (str) and 'frame' (numpy.ndarray).

        This detector detects objects via YOLOv8m and YOLOv8n models. The both of them are used to catch object which
        other model may miss.

        - The results are stored in 'self.results'
        """

        for data in self.frames:
            frame = data['frame']

            results_n = self.model_s(frame,verbose=False)
            results_m = self.model_m(frame,verbose=False)

            # model: yolov8n
            for result in results_n:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model_s.names[cls_id]

                    if label in self.target_objects:
                        self.logger.log(Color.YELLOW, f"{Level.INFO.value} External device detected on frame {data['timestamp']}.")
                        self.results.append(
                            {
                                'timestamp':data['timestamp'],
                                'event_type': Flag.EXTERNAL_DEVICE_DETECTED.value
                            }
                        )

                        break

            # model: yolov8m
            for result in results_m:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model_m.names[cls_id]

                    if label in self.target_objects:
                        self.logger.log(Color.YELLOW, f"{Level.INFO.value} External device detected on frame {data['timestamp']}.")
                        self.results.append(
                            {
                                'timestamp': data['timestamp'],
                                'event_type': Flag.EXTERNAL_DEVICE_DETECTED.value
                            }
                        )

                        break


