from .pipeline import Pipeline

class Detection:

    def __init__(self, p: type[Pipeline],videos:list,video_folder_path:str):
        self.pipeline = p(videos,video_folder_path)

    def run(self,log_results:bool = True, save_results:bool = True) -> None:

        self.pipeline.start_detection()

        if log_results:
            self.pipeline.log_detection_results()

        if save_results:
            self.pipeline.save_results()


