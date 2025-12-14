import cv2
import numpy as np
from tqdm import tqdm

def calculate_flow(imageNames):
    outFrames = []
    hsv = np.zeros_like(cv2.imread(f"./footage/frames/{imageNames[0]}"))
    hsv[..., 1] = 255

    for i in tqdm(range(1, len(imageNames))):
        oldImg = cv2.imread(f"./footage/frames/{imageNames[i-1]}")
        oldImg = cv2.cvtColor(oldImg, cv2.COLOR_BGR2GRAY)
        newImg = cv2.imread(f"./footage/frames/{imageNames[i]}")
        newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(oldImg, newImg, None, 0.7, 4, 27, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
 
        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        outFrames.append(bgr)

    frame_size = cv2.imread(f"./footage/frames/{imageNames[0]}").shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    video_wr = cv2.VideoWriter(
        "./footage/_optFlow.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        (frame_size[1], frame_size[0]),
        isColor=True,
    )
    for idx, frame in enumerate(outFrames):
        frame_annot = frame.copy()
        cv2.putText(
            frame_annot,
            f"Frame {idx}",
            (30, 100),
            font,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )
        video_wr.write(frame_annot)
    video_wr.release()


def calc_sparse_flow(imageNames, vid=None):

    flow_imgs = []

    if vid is not None:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print("Error: Could not open video file.")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

    if vid is None:
        oldImg = cv2.imread(f"./footage/frames/{imageNames[0]}")
    else:
        oldImg = frames[0]
    oldImg_gray = cv2.cvtColor(oldImg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./footage/test.jpg", oldImg_gray)
    # h, w = oldImg_gray.shape
    # mask3070 = np.zeros_like(oldImg_gray)
    # cv2.rectangle(mask3070, (round(w*0.25), round(h*0.25)), (round(w*0.75), round(h*0.75)), 255, -1)
    keypoints = cv2.goodFeaturesToTrack(oldImg_gray, mask=None, maxCorners=25, qualityLevel=0.2, minDistance=10, blockSize=5)
    print(keypoints)

    warp_matrices = []

    nFrames = len(imageNames) if vid is None else len(frames)
    for i in tqdm(range(1, nFrames)):
        
        if vid is None:
            newImg = cv2.imread(f"./footage/frames/{imageNames[i]}")
        else:
            newImg = frames[i]
        newImg_gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(oldImg)
        newpoints, st, _ = cv2.calcOpticalFlowPyrLK(
            oldImg_gray, newImg_gray, keypoints, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)
        )
        good_old = keypoints[st == 1]
        good_new = newpoints[st == 1]

        affine = cv2.estimateAffine2D(good_old, good_new)[0]
        warp_matrices.append(affine)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(np.int32)
            c, d = old.ravel().astype(np.int32)
            #print(f"a: {a}, b: {b}, c: {c}, d: {d}")
            mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 2)
            newImg = cv2.circle(newImg, (a, b), 5, (255, 0, 0), -1)

        flow_img = cv2.add(newImg, mask)
        flow_imgs.append(flow_img)

        oldImg_gray = newImg_gray.copy()
        keypoints = good_new.reshape(-1, 1, 2)
    
    frame_size = flow_imgs[0].shape
    video_wr = cv2.VideoWriter("./footage/_lkFlow.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=True)
    for idx, frame in enumerate(flow_imgs):
        frame_annot = frame.copy()
        cv2.putText(frame_annot, f"Frame {idx}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        video_wr.write(frame_annot)
    video_wr.release()

    return warp_matrices





def stabilize(imageNames):

    def calc_warp_mtx(prevImg, nextImg):
        prevImg = prevImg.copy().astype(np.float32)
        prevImg = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        nextImg = nextImg.copy().astype(np.float32)
        nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
        warp_matrix=np.eye(2, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=prevImg,inputImage=nextImg,
                                        warpMatrix=warp_matrix, motionType=cv2.MOTION_AFFINE, 
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-5))[1]
        return warp_matrix

    def create_warp_stack(images):
        warp_stack = []
        for i, img in tqdm(enumerate(images[:-1])):
            warp_stack += [calc_warp_mtx(img, images[i+1])]
        return np.array(warp_stack)
    
    def create_lk_warp_stack(imageNames):
        return np.array(calc_sparse_flow(imageNames))

    def calc_homography_mtx(warp_stack):
        H_cumul = np.eye(3)
        warps = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
        for i in range(len(warp_stack)):
            H_cumul = np.matmul(warps[i].T, H_cumul)
            yield np.linalg.inv(H_cumul)

    def get_border_pads(img_shape, warp_stack):
        maxmin = []
        corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
        warp_prev = np.eye(3)
        for warp in warp_stack:
            if warp.shape == (2,3):
                warp = np.concatenate([warp, [[0,0,1]]])
            warp = np.matmul(warp, warp_prev)
            warp_invs = np.linalg.inv(warp)
            new_corners = np.matmul(warp_invs, corners)
            xmax,xmin = new_corners[0].max(), new_corners[0].min()
            ymax,ymin = new_corners[1].max(), new_corners[1].min()
            maxmin += [[ymax,xmax], [ymin,xmin]]
            warp_prev = warp.copy()
        maxmin = np.array(maxmin)
        bottom = maxmin[:,0].max()
        top = maxmin[:,0].min()
        left = maxmin[:,1].min()
        right = maxmin[:,1].max()
        return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])

    def generate_stabilized(images, warp_stack):
        top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
        H = calc_homography_mtx(warp_stack)
        stab_images = []
        for i, img in enumerate(images[1:]):
            H_tot = next(H)+np.array([[0,0,left],[0,0,top],[0,0,0]])
            img_warp=cv2.warpPerspective(img,H_tot,(img.shape[1]+left+right,img.shape[0]+top+bottom))
            stab_images += [img_warp]
        return stab_images
    
    imgs = [cv2.imread(f"./footage/frames/{imgName}") for imgName in imageNames]

    #stack = create_warp_stack(imgs)
    stack = create_lk_warp_stack(imageNames)
    stab_frames = generate_stabilized(imgs, stack)
    frame_size = stab_frames[0].shape
    print(frame_size)

    video_wr = cv2.VideoWriter("./footage/_imgStabLK.avi", cv2.VideoWriter.fourcc(*"MJPG"), 50.0, (frame_size[1], frame_size[0]), isColor=True)
    for idx, frame in enumerate(stab_frames):
        frame_annot = frame.copy()
        cv2.putText(frame_annot, f"Frame {idx}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        video_wr.write(frame_annot)
    video_wr.release()
    


#calculate_flow([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

#stabilize([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

#calc_sparse_flow([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

calc_sparse_flow([], vid="./footage/_erodeDilate.avi")