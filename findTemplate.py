# tried but didn't work (leaving for posterity)

import cv2
from tqdm import tqdm

def findTemplate(imageNames):
    template_gray = cv2.imread("./footage/tableTemplate.png", 0)
    
    masks = []
    for name in tqdm(imageNames):
        frame = cv2.imread(f"./footage/frames/{name}")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = template_gray.shape
        match = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(match)
        other_corner = (max_loc[0] + w, max_loc[1] + h)
        cv2.rectangle(frame, max_loc, other_corner, (0, 0, 255), 2)
        masks.append(frame)
    
    frame_size = masks[0].shape
    video_wr = cv2.VideoWriter("./footage/_templatedTable.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=True)
    for idx, frame in enumerate(masks):
        frame_annot = frame.copy()
        cv2.putText(frame_annot, f"Frame {idx}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        video_wr.write(frame_annot)
    video_wr.release()

    # findTemplate([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])