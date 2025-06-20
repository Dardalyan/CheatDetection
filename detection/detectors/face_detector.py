import cv2 as cv
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from .base_detector import Flag, VideoDetector
from detection.logging import  Color, Level


class FaceDetector(VideoDetector):

    def __init__(self, frames: list[dict]):
        super().__init__(frames)
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True)

    def detect(self):
        """
        Detects objects in provided frames.
        Provided frame data list should be parsed via FrameParser.
        It is highly recommended because the expected frame list must be the list of dictionaries which
        contains 'timestamp' (str) and 'frame' (numpy.ndarray).

        This detector detects objects via 'mediapipe'. It extracts the landmarks on the face and according to the
        movements of head and eyes (gaze) in order to detect.

        It checks whether gaze is in the center or whether head turns to left or right.

        - The results are stored in 'self.results'
        """
        for data in self.frames:
            rgb_format = cv.cvtColor(data['frame'],cv.COLOR_BGR2RGB)
            results = self.mesh.process(rgb_format)

            if not results.multi_face_landmarks:
                self.logger.log(Color.YELLOW, f"{Level.INFO.value} No people is detected on frame {data['timestamp']}.")
                self.results.append(
                    {
                        'timestamp': data['timestamp'],
                        'event_type': Flag.NO_PEOPLE.value
                    }
                )

            elif len(results.multi_face_landmarks) > 1:
                self.logger.log(Color.YELLOW, f"{Level.INFO.value} More than one people are detected on frame {data['timestamp']}.")
                self.results.append(
                    {
                        'timestamp': data['timestamp'],
                        'event_type':Flag.MULTIPLE_PEOPLE.value
                    }
                )

            else:
                face_landmark = results.multi_face_landmarks[0]
                #print(data.get('timestamp'))
                face = Face(face_landmark)

                if face.detect_gaze() is not None or face.detect_head_turn() is not None:
                    self.logger.log(Color.YELLOW, f"{Level.INFO.value} Person might be looking away on frame {data['timestamp']}.")
                    self.results.append(
                        {
                            'timestamp': data['timestamp'],
                            'event_type': Flag.LOOKING_AWAY.value
                        }
                    )

        self.mesh.close()




class Face:

    def __init__(self,face_landmark: landmark_pb2.NormalizedLandmarkList):
        """
        Face object keeping the landmarks which are extracted from a frame.
        :param face_landmark: Extracted normalized landmark list to get boundaries of eyes and the point of iris.
        """
        self.face_landmarks = face_landmark
        self.left_eye = Eye(face_landmark.landmark[33].x, face_landmark.landmark[133].x, face_landmark.landmark[468].x)
        self.right_eye = Eye(face_landmark.landmark[362].x, face_landmark.landmark[263].x, face_landmark.landmark[473].x)

    def detect_gaze(self) -> str | None:
        """
        :return: Returns 'LOOKING_AWAY' string if both the eyes are not in the center . Else returns None.
        """
        left_eye = self.left_eye.check_eye_direction()
        right_eye = self.right_eye.check_eye_direction()


        if left_eye == 0 or right_eye == 0:
            return None
        else:
            return 'LOOKING_AWAY'


    def detect_head_turn(self) -> str | None:
        """
        :return: Returns a string value ("HEAD_RIGHT" or  "HEAD_LEFT") if head is turned. Else returns None.
        """
        left_face = self.face_landmarks.landmark[234].x
        right_face = self.face_landmarks.landmark[454].x
        nose = self.face_landmarks.landmark[1].x

        face_center = (left_face + right_face) / 2
        deviation = nose - face_center

        if deviation > 0.03:
            return "HEAD_RIGHT"
        elif deviation < -0.03:
            return "HEAD_LEFT"
        else:
            return None

class Eye:

    def __init__(self,in_bound:float,out_bound:float,iris:float):
        """
        Eye object to represent the eye's position. It stores:
            - Inner eye boundary
            - Outer eye boundary
            - Iris

        :param in_bound: Position of the inner eye boundary (int).
        :param out_bound: Position of the outer eye boundary (int).
        :param iris: Position of the iris (int).
        """
        self.in_bound  = in_bound
        self.out_bound = out_bound
        self.iris = iris

        #print(f'Side:{self.side}', round(self.in_bound, 3), round(self.iris, 3), round(self.out_bound, 3))

    def check_eye_direction(self) -> int:
        """
        :return: Returns '0' (int) if iris not in the center. Else returns '1' (int).
        """

        tolerance = 0.001

        #print(f'IRIS TOLARETED: {round(self.iris + tolerance,3)} BOUND MEAN: {round((self.out_bound + self.in_bound )/2,3)}')

        iris_tolareted = round(self.iris + tolerance,3)
        mean = round((self.out_bound + self.in_bound )/2,3)

        if iris_tolareted == mean:
            return 0
        else:
            if  iris_tolareted == mean + tolerance or iris_tolareted == mean - tolerance:
                return 0
            else:
                return 1
