import cv2
import numpy as np
import math
from thresholding import erode_dilate, output_video
from tqdm import tqdm


def identify_serve(contours, frame_shape, prev_center=None, frame_idx=None):
    h, w = frame_shape[:2]

    MIN_AREA = 20
    MAX_AREA = 280

    best_score = -1.0
    best_center = None
    best_radius = None
    candidates = []

    for ci, c in enumerate(contours):
        area = float(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        bw_f = float(bw)
        bh_f = float(bh)

        extent = area / (bw_f * bh_f + 1e-6)
        aspect = min(bw_f, bh_f) / (max(bw_f, bh_f) + 1e-6)

        perimeter = float(cv2.arcLength(c, True))
        circularity = 0.0 if perimeter < 1e-6 else (4.0 * math.pi * area / (perimeter * perimeter))

        (x0, y0), radius = cv2.minEnclosingCircle(c)
        radius = float(radius)

        circle_area = math.pi * radius * radius
        area_ratio = area / (circle_area + 1e-6)

        center = np.array([x0, y0], dtype=np.float32)

        dist_prev = None
        if prev_center is not None:
            dist_prev = float(np.linalg.norm(center - prev_center))

        score = 2.4 * circularity + 1.6 * area_ratio + 0.6 * aspect
        if dist_prev is not None:
            score += 1.0 * math.exp(-(dist_prev**2) / (2.0 * (70.0**2)))

        candidates.append(
            {
                "idx": ci,
                "area": area,
                "bw": bw_f,
                "bh": bh_f,
                "extent": extent,
                "aspect": aspect,
                "circ": circularity,
                "area_ratio": area_ratio,
                "cx": float(center[0]),
                "cy": float(center[1]),
                "r": radius,
                "dist_prev": dist_prev,
                "score": float(score),
            }
        )

        if score > best_score:
            best_score = score
            best_center = center
            best_radius = radius

    cand_tuples = [
        (np.array([d["cx"], d["cy"]], dtype=np.float32), d["r"], d["circ"], d["area_ratio"], d["score"])
        for d in candidates
    ]
    return best_center, best_radius, cand_tuples


def identify_rally(
    contours,
    frame_shape,
    prev_center=None,
    prev_vel=None,
    frame_idx=None,
    serve_anchor=None,
    rally_age=None,
    miss_count=0,
    stable_run=0,
    force_reverse=False,
    prev_info=None,
):
    h, w = frame_shape[:2]

    pred_center = None
    pred_center_rev = None
    if prev_center is not None and prev_vel is not None:
        pred_center = (prev_center - prev_vel) if force_reverse else (prev_center + prev_vel)

        if not force_reverse and miss_count >= 3:
            pred_center_rev = prev_center - prev_vel
        else:
            pred_center_rev = None

    # early-rally anchor
    use_anchor = (serve_anchor is not None) and (rally_age is not None) and (rally_age <= 20)
    anchor_center = serve_anchor if use_anchor else None

    MIN_AREA = 40
    MAX_AREA = 1400
    MIN_BBOX = 3
    MIN_EXTENT = 0.12

    gate = 160.0 if miss_count == 0 else 320.0

    ROI_X0, ROI_X1 = 0.10 * w, 0.95 * w
    ROI_Y0, ROI_Y1 = 0.05 * h, 0.95 * h

    rej = {"area": 0, "bbox": 0, "extent": 0, "roi": 0, "pred_gate": 0}

    best_score = -1e18
    best_center = None
    best_box = None
    best_info = None
    candidates = []

    vel_dir = None
    if prev_vel is not None:
        vnorm = float(np.linalg.norm(prev_vel))
        if vnorm > 1e-3:
            vel_dir = prev_vel / vnorm

    prev_bboxA = None
    prev_r = None
    prev_elong = None
    if prev_info is not None:
        if prev_info.get("bbox_area") is not None:
            prev_bboxA = float(prev_info["bbox_area"])
        if prev_info.get("r") is not None:
            prev_r = float(prev_info["r"])
        if prev_info.get("elong") is not None:
            prev_elong = float(prev_info["elong"])

    for ci, c in enumerate(contours):
        area = float(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            rej["area"] += 1
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        bw = float(bw)
        bh = float(bh)
        if bw < MIN_BBOX or bh < MIN_BBOX:
            rej["bbox"] += 1
            continue

        perim = float(cv2.arcLength(c, True))
        circ = 0.0 if perim < 1e-6 else (4.0 * math.pi * area / (perim * perim))

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + 1e-6)

        (_, _), r = cv2.minEnclosingCircle(c)
        r = float(r)
        fill = area / (math.pi * r * r + 1e-6)

        extent = area / (bw * bh + 1e-6)
        if extent < MIN_EXTENT:
            rej["extent"] += 1
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy) = rect[0]
        center = np.array([cx, cy], dtype=np.float32)

        if not (ROI_X0 <= cx <= ROI_X1 and ROI_Y0 <= cy <= ROI_Y1):
            rej["roi"] += 1
            continue

        dist_pred = None
        if pred_center is not None:
            d_main = float(np.linalg.norm(center - pred_center))

            if pred_center_rev is not None:
                d_alt = float(np.linalg.norm(center - pred_center_rev))
                d = min(d_main, d_alt)
            else:
                d = d_main

            if force_reverse:
                d = d_main

            dist_pred = d
            if dist_pred > gate:
                rej["pred_gate"] += 1
                continue

        box = cv2.boxPoints(rect).astype(np.int32)

        elong = max(bw, bh) / (min(bw, bh) + 1e-6)

        align = 0.0
        if vel_dir is not None:
            angle = float(rect[2])
            theta = np.deg2rad(angle)
            rect_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            align = float(abs(np.dot(rect_dir, vel_dir)))

        score = 0.0
        if dist_pred is not None:
            sigma = 140.0 if miss_count == 0 else 220.0
            score += 8.0 * math.exp(-(dist_pred**2) / (2.0 * sigma * sigma))

        score += 1.2 * min(elong / 3.0, 1.5)
        score += 2.0 * align

        bboxA = float(bw * bh)

        logA = None
        logr = None
        loge = None

        if prev_bboxA is not None:
            logA = math.log((bboxA + 1e-6) / (prev_bboxA + 1e-6))
            # score += 1 * math.exp(-(logA * logA) / (2.0 * (0.55 ** 2)))

        if prev_r is not None:
            logr = math.log((r + 1e-6) / (prev_r + 1e-6))
            # score += 1 * math.exp(-(logr * logr) / (2.0 * (0.55 ** 2)))

        if prev_elong is not None:
            loge = math.log((elong + 1e-6) / (prev_elong + 1e-6))
            score += 1 * math.exp(-(loge * loge) / (2.0 * (0.70 ** 2)))

        dist_anchor = None
        if anchor_center is not None:
            dist_anchor = float(np.linalg.norm(center - anchor_center))
            fade = max(0.0, 1.0 - (float(rally_age) / 20.0))
            score += 2.0 * fade * math.exp(-(dist_anchor**2) / (2.0 * 200.0 * 200.0))

        info = {
            "idx": ci,
            "area": float(area),
            "bbox_area": float(bboxA),
            "cx": float(cx),
            "cy": float(cy),
            "bw": float(bw),
            "bh": float(bh),
            "extent": float(extent),
            "elong": float(elong),
            "align": float(align),
            "dist_pred": dist_pred,
            "dist_anchor": dist_anchor,
            "rect": rect,
            "box": box,
            "score": float(score),

            # shape
            "perim": float(perim),
            "circ": float(circ),
            "hull_area": float(hull_area),
            "solidity": float(solidity),
            "r": float(r),
            "fill": float(fill),

            # prev scale
            "prev_bboxA": prev_bboxA,
            "prev_r": prev_r,
            "prev_elong": prev_elong,
            "logA": None if logA is None else float(logA),
            "logr": None if logr is None else float(logr),
            "loge": None if loge is None else float(loge),
        }
        candidates.append(info)

        if score > best_score:
            best_score = score
            best_center = center
            best_box = box
            best_info = info

    # debug print
    if frame_idx is not None and 200 < frame_idx < 300:
        mode = "TRACK" if miss_count < 3 else "REACQ"
        print(
            f"\n[frame {frame_idx}] mode={mode} miss={miss_count} stable={stable_run} "
            f"contours={len(contours)} candidates={len(candidates)} mode={mode}"
        )

        if len(candidates) == 0:
            print("  rejection counts:", rej)
        else:
            if best_info is not None:
                dp = "" if best_info["dist_pred"] is None else f" dist_pred={best_info['dist_pred']:.1f}"
                da = "" if best_info["dist_anchor"] is None else f" dist_anchor={best_info['dist_anchor']:.1f}"
                la = "" if best_info["logA"] is None else f" logA={best_info['logA']:+.2f}"
                lr = "" if best_info["logr"] is None else f" logr={best_info['logr']:+.2f}"
                le = "" if best_info["loge"] is None else f" loge={best_info['loge']:+.2f}"
                print(
                    f"  PICKED cand#{best_info['idx']:03d}: score={best_info['score']:.3f} "
                    f"center=({best_info['cx']:.1f},{best_info['cy']:.1f}) "
                    f"bbox=({best_info['bw']:.0f}x{best_info['bh']:.0f}) bboxA={best_info['bbox_area']:.0f} "
                    f"extent={best_info['extent']:.2f} elong={best_info['elong']:.2f} "
                    f"circ={best_info['circ']:.3f} solid={best_info['solidity']:.3f} fill={best_info['fill']:.3f} r={best_info['r']:.1f} "
                    f"align={best_info['align']:.2f}{dp}{da}{la}{lr}{le}"
                )

            top = sorted(candidates, key=lambda d: d["score"], reverse=True)[:5]
            print("  (top candidates)")
            for d in top:
                dp = "" if d["dist_pred"] is None else f" dist_pred={d['dist_pred']:.1f}"
                da = "" if d["dist_anchor"] is None else f" dist_anchor={d['dist_anchor']:.1f}"
                la = "" if d["logA"] is None else f" logA={d['logA']:+.2f}"
                lr = "" if d["logr"] is None else f" logr={d['logr']:+.2f}"
                le = "" if d["loge"] is None else f" loge={d['loge']:+.2f}"
                print(
                    f"  cand#{d['idx']:03d}: score={d['score']:.3f} "
                    f"center=({d['cx']:.1f},{d['cy']:.1f}) "
                    f"bbox=({d['bw']:.0f}x{d['bh']:.0f}) bboxA={d['bbox_area']:.0f} "
                    f"extent={d['extent']:.2f} elong={d['elong']:.2f} "
                    f"circ={d['circ']:.3f} solid={d['solidity']:.3f} fill={d['fill']:.3f} r={d['r']:.1f} "
                    f"align={d['align']:.2f}{dp}{da}{la}{lr}{le}"
                )

    return best_center, best_box, candidates, best_info


def identify_ball(video_path, contours_by_frame, mode="mask"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    annotated_frames = []

    prev_center = None
    prev_vel = np.array([0.0, 0.0], dtype=np.float32)

    pred_center_state = None

    serve_anchor = None
    rally_age = None
    miss_count = 0

    prev_best_info = None
    prev_best_frame = None

    stable_run = 0
    hit_cooldown = 0
    reverse_search = 0

    frame_idx = 0
    n_ct = len(contours_by_frame)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        if frame_idx >= n_ct:
            break

        contours = contours_by_frame[frame_idx]

        if img.ndim == 2:
            ct_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            ct_img = img.copy()

        phase = "SERVE" if frame_idx < 120 else "RALLY"
        track_mode = "REACQ" if (phase == "RALLY" and miss_count >= 3) else "TRACK"

        cv2.drawContours(ct_img, contours, -1, (255, 0, 0), 2)
        cv2.putText(
            ct_img,
            f"{phase}  miss={miss_count}  hitCD={hit_cooldown}  stable={stable_run}  {track_mode}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if frame_idx < 120:
            center, radius, candidates = identify_serve(
                contours,
                frame_shape=ct_img.shape,
                prev_center=prev_center,
                frame_idx=frame_idx,
            )

            for cand_center, cand_radius, circ, area_ratio, cand_score in candidates:
                cc = (int(cand_center[0]), int(cand_center[1]))
                cv2.circle(ct_img, cc, int(cand_radius), (0, 255, 0), 1)

            if center is None and pred_center_state is not None:
                center = pred_center_state + prev_vel
                radius = 8.0

            if center is not None:
                serve_anchor = center.copy()
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center
                pred_center_state = center.copy()

                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                cv2.circle(ct_img, center_int, int(radius), (0, 255, 255), 2)

        else:
            rally_age = 0 if rally_age is None else rally_age + 1

            if frame_idx == 120:
                prev_vel = np.array([0.0, 0.0], dtype=np.float32)
                miss_count = 0
                prev_best_info = None
                prev_best_frame = None
                stable_run = 0
                hit_cooldown = 0
                reverse_search = 0
                pred_center_state = prev_center.copy() if prev_center is not None else None

            if hit_cooldown > 0:
                hit_cooldown -= 1
                center, box, candidates, best_info = None, None, [], None
                stable_run = 0
                coasting = True
            else:
                pred_for_rally = pred_center_state if pred_center_state is not None else prev_center

                center, box, candidates, best_info = identify_rally(
                    contours,
                    frame_shape=ct_img.shape,
                    prev_center=pred_for_rally,
                    prev_vel=prev_vel,
                    frame_idx=frame_idx,
                    serve_anchor=serve_anchor,
                    rally_age=rally_age,
                    miss_count=miss_count,
                    stable_run=stable_run,
                    force_reverse=(reverse_search > 0),
                    prev_info=prev_best_info,  # <-- NEW
                )

                for d in candidates:
                    cv2.drawContours(ct_img, [d["box"]], -1, (0, 255, 0), 2)

                stable_in = stable_run

                # stability update
                if best_info is not None and best_info["dist_pred"] is not None and best_info["dist_pred"] < 40.0:
                    stable_run += 1
                else:
                    stable_run = 0

                if reverse_search > 0:
                    reverse_search -= 1
                if box is not None:
                    reverse_search = 0

                coasting = False

                if (
                    best_info is not None
                    and prev_best_info is not None
                    and prev_best_frame is not None
                    and best_info["dist_pred"] is not None
                    and stable_in >= 8
                    and (frame_idx - prev_best_frame) <= 2
                ):
                    prev_bb_area = float(prev_best_info["bw"] * prev_best_info["bh"])
                    curr_bb_area = float(best_info["bw"] * best_info["bh"])
                    size_ratio = curr_bb_area / (prev_bb_area + 1e-6)

                    BIG_ABS_AREA = 900.0
                    BIG_RATIO = 6.0
                    MID_DIST = 50.0

                    if (curr_bb_area > BIG_ABS_AREA or size_ratio > BIG_RATIO) and best_info["dist_pred"] > MID_DIST:
                        if frame_idx < 175:
                            print(
                                f"  >>> HIT (bbox explode): curr_bb_area={curr_bb_area:.1f} "
                                f"prev_bb_area={prev_bb_area:.1f} ratio={size_ratio:.2f} "
                                f"dist_pred={best_info['dist_pred']:.1f} (forcing MISS + REACQ)"
                            )
                        hit_cooldown = 6
                        reverse_search = 12
                        center, box = None, None
                        stable_run = 0

                        prev_vel = 0.6 * prev_vel

            # coasting
            if center is None and pred_center_state is not None:
                miss_count += 1
                coasting = True

                step = prev_vel
                if reverse_search > 0:
                    step = -step

                pred_center_state = pred_center_state + step

                center = pred_center_state.copy()
                prev_vel = 0.90 * prev_vel
                box = None
            else:
                miss_count = 0

            # draw + update state
            if center is not None:
                center_int = (int(center[0]), int(center[1]))
                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)

                if not coasting:
                    if prev_center is not None:
                        prev_vel = center - prev_center
                    prev_center = center
                    pred_center_state = center.copy()

                # update prev_best_info every successful detection (keeps scale continuity alive)
                if best_info is not None:
                    prev_best_info = best_info
                    prev_best_frame = frame_idx

                if box is not None:
                    cv2.drawContours(ct_img, [box], -1, (0, 255, 255), 2)

        annotated_frames.append(ct_img)
        frame_idx += 1

    cap.release()

    out_path = "./footage/_findball_withDetections.avi"
    video_wr = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        frame_size,
        isColor=True,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, fr in enumerate(annotated_frames):
        fr2 = fr.copy()
        cv2.putText(fr2, f"Frame {idx}", (30, 100), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        video_wr.write(fr2)
    video_wr.release()


def find_ball2(video_path, min_area=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    masked_grays = []

    contours_by_frame = erode_dilate("./footage/_thresholded.avi", min_area=50)
    prev_lines = []
    prev_ball = (-1, -1, [])
    prev_centered = False

    for i, contours in enumerate(contours_by_frame):
        min_rects = [cv2.minAreaRect(ct) for ct in contours]
        min_boxes = np.int64(np.array([cv2.boxPoints(rect) for rect in min_rects]))
        boxes_filt = {"pass": [], "fail": []}

        contour_edge_frame = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
        passing = {"centers": [], "angles": [], "lines": [], "boxes": []}

        for j, box in enumerate(min_boxes):
            cx, cy = min_rects[j][0]
            boxW, boxH = min_rects[j][1]

            angle = np.deg2rad(min_rects[j][2])
            if boxW < boxH:
                angle += np.pi/2
            l = 100
            pt1, pt2 = (round(cx-l*np.cos(angle)), round(cy-l*np.sin(angle))), (round(cx+l*np.cos(angle)), round(cy+l*np.sin(angle)))
            [vx, vy, x0, y0] = cv2.fitLine(np.array([pt1, pt2]), cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy/(vx+1e-6)
            intercept = y0 - slope*x0
            resid = np.ravel(contours[j][..., 1]) - (slope * np.ravel(contours[j][..., 0]) + intercept)
            mse = np.mean(resid ** 2)

            # check current centers against previous frame's lines
            # if len(prev_lines) > 0:
            #     center = np.array([cx, cy])
            #     line_dists = []
            #     for line in prev_lines:
            #         a = np.array(line[0])
            #         b = np.array(line[1])
            #         proj_param = np.dot(center-a, b-a)/np.dot(b-a,b-a)
            #         if proj_param < 0:
            #             line_dists.append(np.linalg.norm(center-a))
            #         elif proj_param > 1:
            #             line_dists.append(np.linalg.norm(center-b))
            #         elif 0.46 < proj_param < 0.54:
            #             line_dists.append(99999)
            #         else:
            #             [vx2, vy2, x02, y02] = cv2.fitLine(np.array(line), cv2.DIST_L2, 0, 0.01, 0.01)
            #             A, B = vy2, -vx2
            #             C = -(A*x02 + B*y02)
            #             line_dists.append((np.abs(A*cx + B*cy + C)/np.sqrt(A*A + B*B))[0])
                
            #     good_lines = []
            #     for k, dist in enumerate(line_dists):
            #         if dist < 200:
            #             [vx2, vy2, x02, y02] = cv2.fitLine(np.array(prev_lines[k]), cv2.DIST_L2, 0, 0.01, 0.01)
            #             if np.abs(vy2/(vx2+1e-6) - slope) < 1:
            #                 good_lines.append(dist)
                
            #     if len(good_lines) == 0:
            #         continue


            hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)

            mask = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)) 
            cv2.drawContours(mask, [box], -1, 255, cv2.FILLED)
            meanS = cv2.mean(hsv[..., 1], mask=mask)[0]

            masked = cv2.bitwise_and(frames[i], frames[i], mask=mask)
            masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            # masked_coords = np.where(masked_gray != 0)
            # new_vals = cv2.equalizeHist(masked_gray[masked_coords])
            # masked_gray[masked_coords] = np.ravel(new_vals)

            # edges = cv2.Canny(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 40, 100)
            # edges = cv2.bitwise_and(edges, edges, mask=cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3), (1,1))))
            # contour_edge_frame = contour_edge_frame + edges

            dist = -1
            if prev_ball[0] != -1:
                center_point = np.array([0.5*(prev_ball[2][0][0] + prev_ball[2][1][0]), 0.5*(prev_ball[2][0][1] + prev_ball[2][1][1])])
                dist = np.linalg.norm(np.array([cx, cy])-center_point)

            if not (20 < cv2.contourArea(box) < 2000):  # not too big or small
                boxes_filt["fail"].append(box)
            elif boxW > 20 and boxH > 20:  # should be thin
                boxes_filt["fail"].append(box)
            elif meanS > 100:  # too bright or colorful to be a ball
                boxes_filt["fail"].append(box)
            elif np.abs(slope) > 1.7:  # too steep to be the ball
                boxes_filt["fail"].append(box)
            elif mse > 40:  # not close enough to a line / stretched out
                boxes_filt["fail"].append(box)
            # elif (prev_ball[0] != -1) and (dist > 200):
            #     boxes_filt["fail"].append(box)
            else:
                boxes_filt["pass"].append(box)
                cv2.circle(frames[i], np.array([int(cx), int(cy)]), 3, (0, 0, 255), -1)

                cv2.line(frames[i], pt1, pt2, (120,255,120), 1)
                passing["centers"].append(np.array([cx, cy]))
                passing["angles"].append(angle)
                passing["lines"].append([pt1, pt2])
                passing["boxes"].append(box)

        print(f"--- Frame {i}: {len(passing['angles'])} passing ---")

        def filter_passing(passing_dict, indices):
            new_passing = {"centers": [], "angles": [], "lines": [], "boxes": []}
            for i in indices:
                for field in ["centers", "angles", "lines", "boxes"]:
                    new_passing[field].append(passing_dict[field][i])
            return new_passing

        ball = (-1, -1, [])  # center, angle, line

        if len(passing["angles"]) > 0:
            center_pos = []
            for j, cand in enumerate(passing["centers"]):
                center_pos.append(passing["centers"][j][0]/w)
            if len(center_pos) > 0:
                center_pos = np.array(center_pos)
                center_scores = 1 / np.abs(center_pos - 0.5)
                closest_center = np.argmax(center_scores)
                if np.max(center_scores) >= 5:
                    print("setting ball to closest center (score > 5)")
                    ball = (closest_center, passing["angles"][closest_center], passing["lines"][closest_center])
                    prev_centered = True
                else:
                    print("no ball found in center of frame")
                    #prev_centered = False

            if ball[0] == -1 and prev_ball[0] != -1:  # if no ball in center, but we have a previous ball
                # find closest to previous ball center, similar slope
                dist_to_prev = [np.linalg.norm(np.array([0.5*(cand[0][0] + cand[1][0]), 0.5*(cand[0][1] + cand[1][1])]) - center_point) for cand in passing["lines"]]
                min_dist = np.argmin(dist_to_prev)
                #if np.abs(passing["angles"][min_dist] - prev_ball[1]) < np.pi/2:
                #    print("setting ball to closest to prev ball with slope within pi/2")
                #    ball = (min_dist, passing["angles"][min_dist], passing["lines"][min_dist])
                ball = (min_dist, passing["angles"][min_dist], passing["lines"][min_dist])
        else:
            print("nothing left, moving on")

        masked_grays.append(contour_edge_frame)

        prev_lines = passing["lines"]

        cv2.drawContours(frames[i], passing["boxes"], -1, (0,255,0), 2)
        cv2.drawContours(frames[i], boxes_filt["fail"], -1, (150,150,255), 1)
        if ball[0] != -1:
            cv2.drawContours(frames[i], [passing["boxes"][ball[0]]], -1, (0,255,255), 3)
        
        prev_ball = (ball[0], ball[1], ball[2])

    output_video(frames, "_erodeDilateRects", isColor=True)
    output_video(masked_grays, "_maskedGray", isColor=False)
    

#contours_by_frame = erode_dilate("./footage/_thresholded.avi", min_area=50)

# identify_ball("./footage/_erodeDilate.avi", contours_by_frame, mode="mask")

# or

#identify_ball("./footage/_cropped.avi", contours_by_frame, mode="no_mask")

find_ball2("./footage/_cropped.avi")