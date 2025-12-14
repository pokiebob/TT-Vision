import cv2
import numpy as np
import sys


def output_video(images, vid_name, isColor=True):

    frame_size = images[0].shape
    video_wr = cv2.VideoWriter(
        f"./footage/{vid_name}.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        (frame_size[1], frame_size[0]),
        isColor=isColor,
    )
    for idx, frame in enumerate(images):
        frame_annot = frame.copy()
        cv2.putText(
            frame_annot,
            f"Frame {idx}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )
        video_wr.write(frame_annot)
    video_wr.release()


def crop_video(vid, start_x, end_x, start_y, end_y):
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit()

    crops = []
    scaled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # scale fractional coords once (on first valid frame)
        if not scaled and start_x < 1:
            h, w = frame.shape[:2]
            start_x, end_x = start_x * w, end_x * w
            start_y, end_y = start_y * h, end_y * h
            scaled = True

        crops.append(frame[int(start_y):int(end_y), int(start_x):int(end_x)])

    cap.release()
    return crops


def hsv_ball_score_suppress_large(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)

    # base white-ish score
    s_score = np.clip(1.0 - (S / 120.0), 0, 1)
    v_score = np.clip((V - 50.0) / 205.0, 0, 1)# lower floor, wider range
    v_score = np.sqrt(v_score) # gamma boost mid-values (keeps ball bright later)

    base_u8 = (s_score * v_score * 255).astype(np.uint8)

    # estimate large bright regions
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    large = cv2.morphologyEx(base_u8, cv2.MORPH_CLOSE, k)
    large = cv2.GaussianBlur(large, (0, 0), 7)

    # subtract part of the large-region baseline
    out = cv2.subtract(base_u8, large // 2)
    return out


def threshold_imgs(imageNames, images=None):
    bgSub = cv2.createBackgroundSubtractorMOG2()

    score_frames = [] 
    thresholded = []
    fgs = []

    if images is None:
        for imgName in imageNames:
            img = cv2.imread(f"./footage/frames/{imgName}")

            ball_score = hsv_ball_score_suppress_large(img)
            score_frames.append(ball_score)

            _, score_bin = cv2.threshold(ball_score, 55, 255, cv2.THRESH_BINARY)

           
            fg = bgSub.apply(ball_score)
            _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            combined = cv2.bitwise_and(score_bin, fg_bin)

            thresholded.append(combined)
            fgs.append(fg_bin) 
    else:
        for img in images:

            ball_score = hsv_ball_score_suppress_large(img)
            score_frames.append(ball_score)

            _, score_bin = cv2.threshold(ball_score, 55, 255, cv2.THRESH_BINARY)

            fg = bgSub.apply(ball_score)  
            _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            # kills  WTT sign
            combined = cv2.bitwise_and(score_bin, fg_bin)

            thresholded.append(combined)
            fgs.append(fg_bin) 

    output_video(images, "_cropped", isColor=True)
    output_video(score_frames, "_gray", isColor=False) 
    output_video(thresholded, "_thresholded", isColor=False)
    output_video(fgs, "_thresholdedFG", isColor=False)



threshold_imgs([], images=crop_video("./footage/_imgStabLK.avi", 0.25, 0.85, 0.3, 0.7))
