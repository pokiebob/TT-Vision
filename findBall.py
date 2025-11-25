import cv2
import numpy as np
import math

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


identify_ball([f"TH_WCQ_point0.mp4-{i:04d}.png" for i in range(373)])