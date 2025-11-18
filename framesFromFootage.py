import os
import cv2

if not os.path.isdir("./footage/frames"):
    os.mkdir("./footage/frames")

def create_frames(video_name):
    vid = cv2.VideoCapture(f"./footage/{video_name}")
    
    frames = 0
    while True:
        success, img = vid.read()
        if not success:
            break
        cv2.imwrite(f"./footage/frames/{video_name}-{frames:04d}.png", img)
        frames += 1
    
    vid.release()

create_frames("TH_WCQ_point0.mp4")