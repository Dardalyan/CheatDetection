import datetime
import math
import numpy as np
import soundfile as sf
from .base_detector import  AudioDetector, Flag
from detection.logging import  Color, Level


class SpeechDetector(AudioDetector):

    def __init__(self, video_path:str, target_file_name:str,target_folder:str = "audio"):
        super().__init__(video_path, target_file_name,target_folder)
        self.check_list = [
            "Whispering",
            "Conversation",
            "Narration, monologue",
            "Talk",
            "Voice"
        ]
        self.threshold = 1e-04

    def detect(self):
        """
        Detects objects in provided frames.
        Provided frame data list should be parsed via FrameParser.
        It is highly recommended because the expected frame list must be the list of dictionaries which
        contains 'timestamp' (str) and 'frame' (numpy.ndarray).

        This detector detects objects via 'YAMNET' model. According to the check_list which is already defined in the class,
        filters the output of the model and catches the audio frames whose scores are within accepted threshold range.

        - The results are stored in 'self.results'
        """

        waveform, sr = sf.read(self.audio_path, dtype='float32')

        frame_length = int(sr * self.frame_duration)
        total_length = len(waveform)
        num_frames = math.ceil(total_length / frame_length)

        if len(waveform.shape) > 1 and waveform.shape[1] == 2:
            waveform = np.mean(waveform, axis=1)

        if sr != 16000:
            raise ValueError('YAMNet requires 16 kHz audio.')


        for i in range(num_frames):
            start = i * frame_length
            end = start + frame_length
            frame = waveform[start:end]

            if len(frame) < frame_length:
                padding = np.zeros(frame_length - len(frame))
                frame = np.concatenate((frame, padding))

            scores, embeddings, spectrogram = self.model(frame)
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            output = np.argsort(mean_scores)[::-1]
            top_labels = [(self.labels[i], mean_scores[i]) for i in output if self.labels[i] in self.check_list and mean_scores[i] > self.threshold]

            check = False if  len(top_labels) == 0 else True
            timestamp = str(datetime.timedelta(seconds=i * self.frame_duration))


            if check:
                self.logger.log(Color.YELLOW, f"{Level.INFO.value} Speech (possible cheating) is detected on frame {timestamp}.")
                self.results.append({
                    "timestamp": timestamp,
                    "event_type": Flag.SPEECH_DETECTED.value
                })




class TypingDetector(AudioDetector):

    def __init__(self, video_path:str, target_file_name:str,target_folder:str = "audio"):
        super().__init__(video_path, target_file_name,target_folder)
        self.check_list = [
            "Keyboard typing",
            "Keypad tones",
            "Text message",
            "Typing",
            "Computer keyboard"
        ]
        self.threshold = 1e-08

    def detect(self):
        """
        Detects objects in provided frames.
        Provided frame data list should be parsed via FrameParser.
        It is highly recommended because the expected frame list must be the list of dictionaries which
        contains 'timestamp' (str) and 'frame' (numpy.ndarray).

        This detector detects objects via 'YAMNET' model. According to the check_list which is already defined in the class,
        filters the output of the model and catches the audio frames whose scores are within accepted threshold range.

        - The results are stored in 'self.results'
        """

        waveform, sr = sf.read(self.audio_path, dtype='float32')

        frame_length = int(sr * self.frame_duration)
        total_length = len(waveform)
        num_frames = math.ceil(total_length / frame_length)

        if len(waveform.shape) > 1 and waveform.shape[1] == 2:
            waveform = np.mean(waveform, axis=1)

        if sr != 16000:
            raise ValueError('YAMNet requires 16 kHz audio.')

        for i in range(num_frames):
            start = i * frame_length
            end = start + frame_length
            frame = waveform[start:end]

            if len(frame) < frame_length:
                padding = np.zeros(frame_length - len(frame))
                frame = np.concatenate((frame, padding))

            scores, embeddings, spectrogram = self.model(frame)
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            output = np.argsort(mean_scores)[::-1]
            top_labels = [(self.labels[i], mean_scores[i]) for i in output if
                          self.labels[i] in self.check_list and mean_scores[i] > self.threshold]

            check = False if len(top_labels) == 0 else True
            timestamp = str(datetime.timedelta(seconds=i * self.frame_duration))

            if check:
                self.logger.log(Color.YELLOW, f"{Level.INFO.value} Typing (possible cheating) is detected on frame {timestamp}.")
                self.results.append({
                    "timestamp": timestamp,
                    "event_type": Flag.KEYBOARD_TYPING_DETECTED.value
                })


