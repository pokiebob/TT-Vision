import cv2
import numpy as np
import sys

def output_video(images, vid_name, isColor=True):
    frame_size = images[0].shape
    video_wr = cv2.VideoWriter(f"./footage/{vid_name}.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=isColor)
    for idx, frame in enumerate(images):
        frame_annot = frame.copy()
        cv2.putText(frame_annot, f"Frame {idx}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        video_wr.write(frame_annot)
    video_wr.release()


def threshold_imgs(imageNames, images=None):
    bgSub = cv2.createBackgroundSubtractorMOG2()
    grays = []
    thresholded = []
    fgs = []

    if images is None:
        for imgName in imageNames:
            img = cv2.imread(f"./footage/frames/{imgName}")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grays.append(img_gray)
            
            _, thresh = cv2.threshold(img_gray, 120, 0, cv2.THRESH_TOZERO)
            _, thresh = cv2.threshold(thresh, 190, 0, cv2.THRESH_TOZERO_INV)
            thresholded.append(thresh)
            
            fg = bgSub.apply(thresh)
            fgs.append(fg)
    else:
        for img in images:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grays.append(img_gray)
            
            _, thresh = cv2.threshold(img_gray, 120, 0, cv2.THRESH_TOZERO)
            _, thresh = cv2.threshold(thresh, 190, 0, cv2.THRESH_TOZERO_INV)
            thresholded.append(thresh)
            
            fg = bgSub.apply(thresh)
            fgs.append(fg)
    
    output_video(grays, "_gray", isColor=False)
    output_video(thresholded, "_thresholded", isColor=False)
    output_video(fgs, "_thresholdedFG", isColor=False)


def crop_video(vid, start_x, end_x, start_y, end_y):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit()
    
    crops = []
    while True:
        ret, frame = cap.read()
        if start_x < 1:
            h, w = frame.shape[:2]
            start_x, end_x = start_x*w, end_x*w
            start_y, end_y = start_y*h, end_y*h
        if not ret:
            break  # Break the loop if no more frames
        #print(f"cropping x={start_x} to {end_x}, y={start_y} to {end_y}")
        crops.append(frame[int(start_y):int(end_y), int(start_x):int(end_x)])
    
    return crops


#threshold_imgs([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

threshold_imgs([], images=crop_video("./footage/_imgStabLK.avi", 0.25, 0.85, 0.3, 0.7))
