import cv2
import numpy as np
import math
from thresholding import erode_dilate


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

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        (x0, y0), radius = cv2.minEnclosingCircle(c)
        radius = float(radius)

        circle_area = math.pi * radius * radius
        area_ratio = area / (circle_area + 1e-6)

        center = np.array([x0, y0], dtype=np.float32)

        # if not (0.2 * w <= center[0] <= 0.8 * w and 0.2 * h <= center[1] <= 0.7 * h):
        #     continue

        dist_prev = None
        if prev_center is not None:
            dist_prev = float(np.linalg.norm(center - prev_center))

        score = 2.4 * circularity + 1.6 * area_ratio + 0.6 * aspect
        if dist_prev is not None:
            score += 1.0 * math.exp(-(dist_prev**2) / (2.0 * (70.0**2)))


        candidates.append(
            {"idx": ci, "area": area, "bw": bw_f, "bh": bh_f, "extent": extent,
             "aspect": aspect, "circ": circularity, "area_ratio": area_ratio,
             "cx": float(center[0]), "cy": float(center[1]), "r": radius,
             "dist_prev": dist_prev, "score": float(score)}
        )

        if score > best_score:
            best_score = score
            best_center = center
            best_radius = radius

    # if frame_idx is not None:
    #     print(f"\n[frame {frame_idx}] serve contours={len(contours)} candidates={len(candidates)}")

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
):
    h, w = frame_shape[:2]

    # --- prediction ---
    pred_center = None
    pred_center_rev = None
    if prev_center is not None and prev_vel is not None:
        pred_center = prev_center + prev_vel
        if miss_count >= 3:
            pred_center_rev = prev_center - 1.0 * prev_vel  # allow opposite direction in REACQ

    # early-rally anchor (optional)
    use_anchor = (serve_anchor is not None) and (rally_age is not None) and (rally_age <= 20)
    anchor_center = serve_anchor if use_anchor else None

    # --- light filters ---
    MIN_AREA = 15
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
            d_fwd = float(np.linalg.norm(center - pred_center))
            d = d_fwd
            if pred_center_rev is not None:
                d_rev = float(np.linalg.norm(center - pred_center_rev))
                d = min(d_fwd, d_rev)
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
            score += 3.0 * math.exp(-(dist_pred**2) / (2.0 * sigma * sigma))

        score += 1.2 * min(elong / 3.0, 1.5)
        score += 3.5 * align

        dist_anchor = None
        if anchor_center is not None:
            dist_anchor = float(np.linalg.norm(center - anchor_center))
            fade = max(0.0, 1.0 - (float(rally_age) / 20.0))
            score += 2.0 * fade * math.exp(-(dist_anchor**2) / (2.0 * 200.0 * 200.0))

        info = {
            "idx": ci,
            "area": area,
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
        }
        candidates.append(info)

        if score > best_score:
            best_score = score
            best_center = center
            best_box = box
            best_info = info

    # debug print
    if frame_idx is not None and frame_idx < 175:
        mode = "TRACK" if miss_count < 3 else "REACQ"
        print(f"\n[frame {frame_idx}] mode={mode} miss={miss_count} stable={stable_run} contours={len(contours)} candidates={len(candidates)} mode={mode}")

        if len(candidates) == 0:
            print("  rejection counts:", rej)
        else:
            if best_info is not None:
                dp = "" if best_info["dist_pred"] is None else f" dist_pred={best_info['dist_pred']:.1f}"
                da = "" if best_info["dist_anchor"] is None else f" dist_anchor={best_info['dist_anchor']:.1f}"
                print(
                    f"  PICKED cand#{best_info['idx']:03d}: score={best_info['score']:.3f} "
                    f"center=({best_info['cx']:.1f},{best_info['cy']:.1f}) "
                    f"bbox=({best_info['bw']:.0f}x{best_info['bh']:.0f}) "
                    f"extent={best_info['extent']:.2f} elong={best_info['elong']:.2f} "
                    f"align={best_info['align']:.2f}{dp}{da}"
                )

            top = sorted(candidates, key=lambda d: d["score"], reverse=True)[:5]
            print("  (top candidates)")
            for d in top:
                dp = "" if d["dist_pred"] is None else f" dist_pred={d['dist_pred']:.1f}"
                da = "" if d["dist_anchor"] is None else f" dist_anchor={d['dist_anchor']:.1f}"
                print(
                    f"  cand#{d['idx']:03d}: score={d['score']:.3f} "
                    f"center=({d['cx']:.1f},{d['cy']:.1f}) "
                    f"elong={d['elong']:.2f} align={d['align']:.2f}{dp}{da}"
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

    serve_anchor = None
    rally_age = None
    miss_count = 0

    prev_best_info = None
    prev_best_frame = None

    stable_run = 0 
    hit_cooldown = 0

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

            if center is None and prev_center is not None:
                center = prev_center + prev_vel
                radius = 8.0

            if center is not None:
                serve_anchor = center.copy()
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

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

            if hit_cooldown > 0:
                hit_cooldown -= 1
                center, box, candidates, best_info = None, None, [], None
                miss_count = max(miss_count, 3)
                stable_run = 0
            else:
                center, box, candidates, best_info = identify_rally(
                    contours,
                    frame_shape=ct_img.shape,
                    prev_center=prev_center,
                    prev_vel=prev_vel,
                    frame_idx=frame_idx,
                    serve_anchor=serve_anchor,
                    rally_age=rally_age,
                    miss_count=miss_count,
                    stable_run=stable_run,
                )

                for d in candidates:
                    cv2.drawContours(ct_img, [d["box"]], -1, (0, 255, 0), 2)

                # stability
                if best_info is not None and best_info["dist_pred"] is not None and best_info["dist_pred"] < 40.0:
                    stable_run += 1
                else:
                    stable_run = 0

                # HIT OVERRIDE 
                if (
                    center is not None
                    and best_info is not None
                    and prev_best_info is not None
                    and prev_best_frame is not None
                    and rally_age is not None
                    and rally_age >= 12 
                    and stable_run >= 3
                    and (frame_idx - prev_best_frame) <= 3 
                    and best_info["dist_pred"] is not None
                ):
                    prev_c = np.array([prev_best_info["cx"], prev_best_info["cy"]], dtype=np.float32)
                    jump = float(np.linalg.norm(center - prev_c))

                    prev_bb_area = float(prev_best_info["bw"] * prev_best_info["bh"])
                    curr_bb_area = float(best_info["bw"] * best_info["bh"])
                    size_ratio = max(
                        curr_bb_area / (prev_bb_area + 1e-6),
                        (prev_bb_area + 1e-6) / (curr_bb_area + 1e-6),
                    )

                    # dynamic threshold based on speed
                    vnorm = float(np.linalg.norm(prev_vel)) if prev_vel is not None else 0.0
                    jump_thr = max(95.0, 3.0 * vnorm + 40.0)

                    gate = 160.0 if miss_count == 0 else 320.0
                    far_from_pred = best_info["dist_pred"] > 0.70 * gate

                    HIT_SIZE_RATIO = 3.0

                    if (jump > jump_thr) and far_from_pred and (size_ratio > HIT_SIZE_RATIO):
                        if frame_idx < 175:
                            print(
                                f"  >>> HIT OVERRIDE: jump={jump:.1f}px (thr={jump_thr:.1f}) "
                                f"dist_pred={best_info['dist_pred']:.1f} size_ratio={size_ratio:.2f} "
                                f"(forcing MISS + REACQ)"
                            )
                        hit_cooldown = 4
                        center, box = None, None
                        miss_count = max(miss_count, 3)
                        stable_run = 0

            # coast if missing
            if center is None and prev_center is not None:
                miss_count += 1
                center = prev_center + prev_vel
                prev_vel = 0.90 * prev_vel
                box = None
            else:
                miss_count = 0

            # draw + update state
            if center is not None:
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                if box is not None:
                    cv2.drawContours(ct_img, [box], -1, (0, 255, 255), 2)

                    prev_best_info = best_info
                    prev_best_frame = frame_idx

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


contours_by_frame = erode_dilate("./footage/_thresholded.avi", min_area=50)

# identify_ball("./footage/_erodeDilate.avi", contours_by_frame, mode="mask")

# or

identify_ball("./footage/_cropped.avi", contours_by_frame, mode="no_mask")
