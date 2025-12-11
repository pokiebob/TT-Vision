import cv2
import numpy as np
import math
from tqdm import tqdm

def identify_serve(contours, frame_shape, hsv_img=None, prev_center=None, frame_idx=None):
    h, w = frame_shape[:2]

    MIN_AREA = 20
    MAX_AREA = 280
    MIN_RADIUS = 6
    MAX_RADIUS = 12

    best_score = -1.0
    best_center = None
    best_radius = None
    best_hsv = None

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue

        circle_area = math.pi * radius * radius
        area_ratio = area / (circle_area + 1e-6)

        center = np.array([x, y], dtype=np.float32)

        # spatial filters: serve will only occur around the table
        if center[1] < .3 * h or center[1] > .7 * h:
            continue
        if center[0] < .3 * w or center[0] > .7 * w:
            continue

        score = 2.0 * circularity + 1.0 * area_ratio

        # tracking prior: prefer close to previous center
        if prev_center is not None:
            dist = float(np.linalg.norm(center - prev_center))
            score += 1.0 * math.exp(- (dist ** 2) / (2.0 * (40.0 ** 2)))

        candidates.append((center, radius, circularity, area_ratio, score))

        # compute local HSV to filter by V
        hsv_mean = None
        if hsv_img is not None:
            cy = int(round(center[1]))
            cx = int(round(center[0]))
            cy0 = max(cy - 1, 0)
            cy1 = min(cy + 2, hsv_img.shape[0])
            cx0 = max(cx - 1, 0)
            cx1 = min(cx + 2, hsv_img.shape[1])
            patch = hsv_img[cy0:cy1, cx0:cx1]
            if patch.size > 0:
                hsv_mean = patch.mean(axis=(0, 1))

        if hsv_mean[2] < 200:
            continue

        if score > best_score:
            best_score = score
            best_center = center
            best_radius = float(radius)
            best_hsv = hsv_mean

    if best_center is None:
        print(f"[frame {frame_idx}] NO ball picked (0 candidates or all filtered)\n")
    else:
        print(
            f"[frame {frame_idx}] PICKED ball at ({best_center[0]:.1f},{best_center[1]:.1f}), "
            f"r={best_radius:.2f}, score={best_score:.3f}, "
            f"HSV≈({best_hsv[0]:.1f}, {best_hsv[1]:.1f}, {best_hsv[2]:.1f})\n"
        )

    return best_center, best_radius, candidates


def identify_ball(imageNames):
    # -> initialize subtractor and containers
    bgSub = cv2.createBackgroundSubtractorMOG2()
    fgs = []
    fgsThresh = []
    cts = []

    prev_center = None
    prev_vel = np.array([0.0, 0.0], dtype=np.float32)

    # -> process frames
    for frame_idx, imgName in enumerate(imageNames):
        img = cv2.imread(f"./footage/frames/{imgName}")
        if img is None:
            continue

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # background substitution
        fg = bgSub.apply(img)
        fgs.append(fg)
        
        # find and render contours on foreground and binary foreground
        _, thresh = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
        fgsThresh.append(thresh)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ct_img = img.copy()
        cv2.drawContours(ct_img, contours, -1, (0, 255, 0), 2)

        # assume serve occurs only in first 120 frames
        if frame_idx < 120:
            center, radius, candidates = identify_serve(
                contours,
                frame_shape=img.shape,
                hsv_img=hsv_img,
                prev_center=prev_center,
                frame_idx=frame_idx,
            )

            # draw all candidates as blue circles 
            for cand_center, cand_radius, circ, area_ratio, cand_score in candidates:
                cc = (int(cand_center[0]), int(cand_center[1]))
                cv2.circle(ct_img, cc, int(cand_radius), (255, 0, 0), 2)

            if center is None and prev_center is not None:
                center = prev_center + prev_vel
                radius = 8.0  

            if center is not None:
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

                # red dot + yellow circle for chosen ball
                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                cv2.circle(ct_img, center_int, int(radius), (0, 255, 255), 2)

        cts.append(ct_img)

    # p> write videos
    frame_size = cv2.imread(f"./footage/frames/{imgName}").shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    video_wr = cv2.VideoWriter(
        "./footage/_extractedFG.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        (frame_size[1], frame_size[0]),
        isColor=False,
    )
    for idx, fg in enumerate(fgs):
        fg_annot = fg.copy()
        cv2.putText(
            fg_annot,
            f"Frame {idx}",
            (30, 60),
            font,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )
        video_wr.write(fg_annot)
    video_wr.release()

    video_th_wr = cv2.VideoWriter(
        "./footage/_extractedFGthresh.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        (frame_size[1], frame_size[0]),
        isColor=False,
    )
    for idx, fgT in enumerate(fgsThresh):
        fgT_annot = fgT.copy()
        cv2.putText(
            fgT_annot,
            f"Frame {idx}",
            (30, 60),
            font,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )
        video_th_wr.write(fgT_annot)
    video_th_wr.release()

    video_ct_wr = cv2.VideoWriter(
        "./footage/_withContours.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        (frame_size[1], frame_size[0]),
    )
    for idx, ct in enumerate(cts):
        ct_annot = ct.copy()
        cv2.putText(
            ct_annot,
            f"Frame {idx}",
            (30, 60),
            font,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        video_ct_wr.write(ct_annot)
    video_ct_wr.release()


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
            (30, 60),
            font,
            1.0,
            255,
            2,
            cv2.LINE_AA,
        )
        video_wr.write(frame_annot)
    video_wr.release()


def calc_sparse_flow(imageNames):

    flow_imgs = []

    oldImg = cv2.imread(f"./footage/frames/{imageNames[0]}")
    oldImg_gray = cv2.cvtColor(oldImg, cv2.COLOR_BGR2GRAY)
    h, w = oldImg_gray.shape
    mask3070 = np.zeros_like(oldImg_gray)
    cv2.rectangle(mask3070, (round(w*0.25), round(h*0.25)), (round(w*0.75), round(h*0.75)), 255, -1)
    keypoints = cv2.goodFeaturesToTrack(oldImg_gray, mask=mask3070, maxCorners=25, qualityLevel=0.5, minDistance=10, blockSize=5)

    warp_matrices = []

    for i in tqdm(range(1, len(imageNames))):
        
        newImg = cv2.imread(f"./footage/frames/{imageNames[i]}")
        newImg_gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(oldImg)
        newpoints, st, _ = cv2.calcOpticalFlowPyrLK(
            oldImg_gray, newImg_gray, keypoints, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)
        )
        good_old = keypoints[st == 1]
        good_new = newpoints[st == 1]

        affine = cv2.estimateAffine2D(good_old, good_new)[0]
        hom = cv2.findHomography(good_old, good_new)[0]
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
        cv2.putText(frame_annot, f"Frame {idx}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
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
        cv2.putText(frame_annot, f"Frame {idx}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        video_wr.write(frame_annot)
    video_wr.release()
    

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



# -------

#identify_ball([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

#calculate_flow([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

stabilize([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

#findTemplate([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])

#calc_sparse_flow([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])