from detection import Detection,CheatDetectionPipeline

video_list = ['interview1.mov','interview2.mov','interview3.mov','interview4.mov','interview5.mov']

dt = Detection(p= CheatDetectionPipeline,videos=video_list,video_folder_path='assets/videos')
dt.run(log_results=True, save_results=True)

