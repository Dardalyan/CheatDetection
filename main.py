from detection import Detection,CheatDetectionPipeline

# The names of videos under the folder assets/videos
# These are placeholder .mov files and do not contain actual interview footage.
# Please replace them with your own videos for proper testing.
video_list = ['interview1.mov','interview2.mov','interview3.mov','interview4.mov','interview5.mov']

dt = Detection(p= CheatDetectionPipeline,videos=video_list,video_folder_path='assets/videos')
dt.run(log_results=True, save_results=True)

