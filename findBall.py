import cv2
import numpy as np
import math
# from thresholding import threshold_imgs

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

def identify_rally(
    contours,
    frame_shape,
    hsv_img=None,
    prev_center=None,
    prev_vel=None,
    frame_idx=None,
    serve_anchor=None,
    rally_age=None,
):
    h, w = frame_shape[:2]

    pred_center = None
    if prev_center is not None and prev_vel is not None:
        pred_center = prev_center + prev_vel

    # serve-anchor prior
    use_anchor = (serve_anchor is not None) and (rally_age is not None) and (rally_age <= 20)
    anchor_center = serve_anchor if use_anchor else None

    candidates = []

    MIN_AREA = 20
    MAX_AREA = 1000

    # reject super-tall blobs
    MAX_BH_BW = 0.9  

    best_score = -1e18
    best_center = None
    best_box = None
    best_info = None

    for ci, c in enumerate(contours):
        area = float(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if area <= 0:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        bw = float(bw)
        bh = float(bh)
        bh_bw = bh / (bw + 1e-6)

        if bw > 0 and bh_bw > MAX_BH_BW:
            continue

        # oriented rect 
        rect = cv2.minAreaRect(c)  # ((cx,cy),(wr,hr),angle)
        (cx, cy) = rect[0]
        (wr, hr) = rect[1]
        wr = float(wr)
        hr = float(hr)

        center = np.array([cx, cy], dtype=np.float32)

        if center[1] < 0.3 * h or center[1] > 0.7 * h:
            continue

        # HSV patch 
        hsv_mean = None
        H = S = V = None
        if hsv_img is not None:
            iy = int(round(center[1]))
            ix = int(round(center[0]))
            y0 = max(iy - 2, 0)
            y1 = min(iy + 3, hsv_img.shape[0])
            x0 = max(ix - 2, 0)
            x1 = min(ix + 3, hsv_img.shape[1])
            patch = hsv_img[y0:y1, x0:x1]
            if patch.size > 0:
                hsv_mean = patch.mean(axis=(0, 1))
                H, S, V = float(hsv_mean[0]), float(hsv_mean[1]), float(hsv_mean[2])

        # distance to prediction
        dist_pred = None
        if pred_center is not None:
            dist_pred = float(np.linalg.norm(center - pred_center))

        # distance to serve anchor (early rally)
        dist_anchor = None
        if anchor_center is not None:
            dist_anchor = float(np.linalg.norm(center - anchor_center))

        box = cv2.boxPoints(rect).astype(np.int32)

        score = 0.0
        if V is not None:
            score += 0.01 * V                 # prefer brighter

        # prefer near predicted location (if available)
        if dist_pred is not None:
            score += 2.0 * math.exp(-(dist_pred * dist_pred) / (2.0 * (120.0 ** 2)))


        # prefer near serve anchor for first ~20 rally frames (strong early, fades out)
        if dist_anchor is not None and rally_age is not None:
            sigma = 180.0 # "launch zone" radius
            fade = max(0.0, 1.0 - (rally_age / 20.0))
            score += (4.0 * fade) * math.exp(-(dist_anchor * dist_anchor) / (2.0 * sigma * sigma))

        # store candidate info f
        info = {
            "idx": ci,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
            "bw": bw,
            "bh": bh,
            "bh_bw": float(bh_bw),
            "wr": wr,
            "hr": hr,
            "H": H,
            "S": S,
            "V": V,
            "dist_pred": dist_pred,
            "dist_anchor": dist_anchor,
            "rect": rect,
            "box": box,
            "score": score,
        }
        candidates.append(info)

        if score > best_score:
            best_score = score
            best_center = center
            best_box = box
            best_info = info

    if frame_idx < 180 or frame_idx > 210:
        return best_center, best_box, candidates
    print(f"\n[frame {frame_idx}] rally contours={len(contours)} candidates={len(candidates)}")
    if pred_center is not None and prev_center is not None and prev_vel is not None:
        print(
            f"  pred=({pred_center[0]:.1f},{pred_center[1]:.1f}) "
            f"prev=({prev_center[0]:.1f},{prev_center[1]:.1f}) "
            f"v=({prev_vel[0]:.1f},{prev_vel[1]:.1f})"
        )
    if anchor_center is not None and rally_age is not None:
        print(f"  serve_anchor=({anchor_center[0]:.1f},{anchor_center[1]:.1f}) rally_age={rally_age}")

    cand_to_print = sorted(candidates, key=lambda d: d["score"], reverse=True)[:10]
    print(f"  (printing top 10 by score; total candidates={len(candidates)})")


    for d in cand_to_print:
        if d["V"] is None:
            hsv_str = "HSV=None"
        else:
            hsv_str = f"HSV=({d['H']:.1f},{d['S']:.1f},{d['V']:.1f})"

        dist_pred_str = "" if d["dist_pred"] is None else f" dist_pred={d['dist_pred']:.1f}"
        dist_anch_str = "" if d["dist_anchor"] is None else f" dist_anchor={d['dist_anchor']:.1f}"

        print(
            f"  cand#{d['idx']:03d}: area={d['area']:.1f} "
            f"center=({d['cx']:.1f},{d['cy']:.1f}) "
            f"bbox=(w={d['bw']:.1f},h={d['bh']:.1f},h/w={d['bh_bw']:.2f}) "
            f"{hsv_str}{dist_pred_str}{dist_anch_str}"
        )

    if best_center is None:
        print(f"[frame {frame_idx}] NO pick (no candidates)\n")
        return None, None, candidates

    print(
        f"[frame {frame_idx}] PICKED idx={best_info['idx']} center=({best_info['cx']:.1f},{best_info['cy']:.1f}) "
        f"area={best_info['area']:.1f} "
        f"V={best_info['V'] if best_info['V'] is not None else None} score={best_score:.3f}\n"
    )

    return best_center, best_box, candidates


def identify_ball(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 60.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    # -> initialize subtractor and containers
    bgSub = cv2.createBackgroundSubtractorMOG2()
    fgs = []
    fgsThresh = []
    cts = []

    prev_center = None
    prev_vel = np.array([0.0, 0.0], dtype=np.float32)

    serve_anchor = None
    rally_age = None


    # -> process frames
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

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
                serve_anchor = center.copy() 
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

                # red dot + yellow circle for chosen ball
                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                cv2.circle(ct_img, center_int, int(radius), (0, 255, 255), 2)
        else:
            if rally_age is None:
                rally_age = 0
            else:
                rally_age += 1

            center, box, candidates = identify_rally(
                contours,
                frame_shape=img.shape,
                hsv_img=hsv_img,
                prev_center=prev_center,
                prev_vel=prev_vel,
                frame_idx=frame_idx,
                serve_anchor=serve_anchor,
                rally_age=rally_age,
            )

            # draw all candidates as blue boxes
            for d in candidates:
                cv2.drawContours(ct_img, [d["box"]], -1, (255, 0, 0), 2)


            if center is None and prev_center is not None:
                center = prev_center + prev_vel
                # assume constant velocity
            if center is not None:
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

                # red dot + yellow box for chosen ball
                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                if box is not None:
                    cv2.drawContours(ct_img, [box], -1, (0, 255, 255), 2)
            

        cts.append(ct_img)
        frame_idx += 1

    # -> write videos
    # frame_size = cv2.imread(f"./footage/frames/{imgName}").shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    video_wr = cv2.VideoWriter(
        "./footage/_extractedFG.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        frame_size,
        isColor=False,
    )
    for idx, fg in enumerate(fgs):
        fg_annot = fg.copy()
        cv2.putText(
            fg_annot,
            f"Frame {idx}",
            (30, 100),
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
        frame_size,
        isColor=False,
    )
    for idx, fgT in enumerate(fgsThresh):
        fgT_annot = fgT.copy()
        cv2.putText(
            fgT_annot,
            f"Frame {idx}",
            (30, 100),
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
        frame_size,
    )
    for idx, ct in enumerate(cts):
        ct_annot = ct.copy()
        cv2.putText(
            ct_annot,
            f"Frame {idx}",
            (30, 100),
            font,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        video_ct_wr.write(ct_annot)
    video_ct_wr.release()




# -------


# identify_ball("./footage/_cropped.avi")
identify_ball("./footage/_thresholded.avi")
# identify_ball("./footage/_thresholdedFG.avi")

