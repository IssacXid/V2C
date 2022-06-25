IIT-V2C Dataset

- Folder "avi_video": contains video from the original Breakfast dataset [1]
- File "train.txt": contains video_id, frame_id, and groundtruth command for each sub-video used for training
- File "test.txt": contains video_id, frame_id, and groundtruth command for each sub-video used for testing

Note 1:
Each sub-video in file "train.txt" or "test.txt" is formatted as follows:
- line0: sub-video-id (e.g., P28_cam02_P28_sandwich_12)
- line1: starting_frame ending_frame (e.g., 938 1015  ---> starting_frame is 938, ending_frame is 1015)
- line2: command sentence groundtruth (e.g., righthand scoop butter)
- line3: empty line

Note 2: 
- The frames are extracted from each video in "avi_video" folder with framerate is 15 frame per second. The index of the first frame is 0.

Note 3:
- We uniformly sample 30 frames from each sub-video (the ending_frame from each action is not selected). 
- If there are not enough 30 frames in the sub-video, we use an artificial mean frame from ImageNet dataset (details in the paper).

---------------------------------
###### Contact: anh.nguyen@iit.it


[1] H. Kuehne, A. B. Arslan and T. Serre. The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities. CVPR, 2014
