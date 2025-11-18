import cv2
#print(cv2.getBuildInformation())

def identify_ball(imageNames):
    bgSub = cv2.createBackgroundSubtractorMOG2()
    fgs = []
    fgsThresh = []
    cts = []
    for imgName in imageNames:
        img = cv2.imread(f"./footage/frames/{imgName}")
        # background substitution
        fgs.append(bgSub.apply(img))
        # find and render contours
        _, thresh = cv2.threshold(fgs[-1], 180, 255, cv2.THRESH_BINARY)
        fgsThresh.append(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cts.append(cv2.drawContours(img, contours, -1, (0, 255, 0), 2))



    frame_size = cv2.imread(f"./footage/frames/{imgName}").shape[:2]
    video_wr = cv2.VideoWriter("./footage/_extractedFG.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=False)
    for fg in fgs:
        video_wr.write(fg)
    video_wr.release()

    video_th_wr = cv2.VideoWriter("./footage/_extractedFGthresh.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=False)
    for fgT in fgsThresh:
        video_th_wr.write(fgT)
    video_th_wr.release()

    video_ct_wr = cv2.VideoWriter("./footage/_withContours.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]))
    for ct in cts:
        video_ct_wr.write(ct)
    video_ct_wr.release()



identify_ball([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])
