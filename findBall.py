import cv2
import numpy as np
import math


def identify_serve(contours, frame_shape, prev_center=None, frame_idx=None):
    h, w = frame_shape[:2]

    MIN_AREA = 20
    MAX_AREA = 280
    MIN_RADIUS = 6
    MAX_RADIUS = 12

    # anti-grain
    MIN_BBOX = 4  # reject tiny specks
    MIN_EXTENT = 0.25

    best_score = -1.0
    best_center = None
    best_radius = None

    candidates = []

    for ci, c in enumerate(contours):
        area = float(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < MIN_BBOX or bh < MIN_BBOX:
            continue

        extent = area / (float(bw) * float(bh) + 1e-6)
        if extent < MIN_EXTENT:
            continue

        perimeter = float(cv2.arcLength(c, True))
        if perimeter == 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        (x0, y0), radius = cv2.minEnclosingCircle(c)
        radius = float(radius)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue

        circle_area = math.pi * radius * radius
        area_ratio = area / (circle_area + 1e-6)

        center = np.array([x0, y0], dtype=np.float32)

        # if not (0.2 * w <= center[0] <= 0.8 * w and 0.2 * h <= center[1] <= 0.7 * h):
        #     continue

        score = 2.0 * circularity + 1.0 * area_ratio

        # tracking prior
        dist_prev = None
        if prev_center is not None:
            dist_prev = float(np.linalg.norm(center - prev_center))
            score += 1.0 * math.exp(-(dist_prev**2) / (2.0 * (40.0**2)))

        candidates.append(
            {
                "idx": ci,
                "area": area,
                "bw": float(bw),
                "bh": float(bh),
                "extent": float(extent),
                "circ": float(circularity),
                "area_ratio": float(area_ratio),
                "cx": float(center[0]),
                "cy": float(center[1]),
                "r": float(radius),
                "dist_prev": dist_prev,
                "score": float(score),
            }
        )

        if score > best_score:
            best_score = score
            best_center = center
            best_radius = radius

    if frame_idx is not None:
        print(f"\n[frame {frame_idx}] serve contours={len(contours)} candidates={len(candidates)}")
        top = sorted(candidates, key=lambda d: d["score"], reverse=True)[:10]
        print("  (top 10 by score)")
        for d in top:
            dist_str = "" if d["dist_prev"] is None else f" dist_prev={d['dist_prev']:.1f}"
            print(
                f"  cand#{d['idx']:03d}: score={d['score']:.3f} "
                f"center=({d['cx']:.1f},{d['cy']:.1f}) r={d['r']:.1f} "
                f"area={d['area']:.1f} bbox=({d['bw']:.0f}x{d['bh']:.0f}) "
                f"extent={d['extent']:.2f} circ={d['circ']:.2f} aratio={d['area_ratio']:.2f}"
                f"{dist_str}"
            )

        if best_center is None:
            print(f"[frame {frame_idx}] NO ball picked\n")
        else:
            print(
                f"[frame {frame_idx}] PICKED ball at ({best_center[0]:.1f},{best_center[1]:.1f}), "
                f"r={best_radius:.2f}, score={best_score:.3f}\n"
            )

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
):
    h, w = frame_shape[:2]

    pred_center = None
    if prev_center is not None and prev_vel is not None:
        pred_center = prev_center + prev_vel

    use_anchor = (serve_anchor is not None) and (rally_age is not None) and (rally_age <= 20)
    anchor_center = serve_anchor if use_anchor else None

    # filters
    MIN_AREA = 20
    MAX_AREA = 1000
    MAX_BH_BW = 0.9

    # anti-grain
    MIN_BBOX = 4
    MIN_EXTENT = 0.18  # rally can be more distorted

    best_score = -1e18
    best_center = None
    best_box = None
    best_info = None
    candidates = []

    for ci, c in enumerate(contours):
        area = float(cv2.contourArea(c))
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        bw = float(bw)
        bh = float(bh)

        # kill tiny grain specks
        if bw < MIN_BBOX or bh < MIN_BBOX:
            continue

        extent = area / (bw * bh + 1e-6)
        if extent < MIN_EXTENT:
            continue

        bh_bw = bh / (bw + 1e-6)
        if bw > 0 and bh_bw > MAX_BH_BW:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy) = rect[0]
        center = np.array([cx, cy], dtype=np.float32)

        # tighter search window right after serve (first ~30 rally frames)
        # if rally_age is not None and rally_age <= 30:
        #     if not (0.30 * w <= center[0] <= 0.70 * w):
        #         continue

        # # table band constraint
        # if center[1] < 0.3 * h or center[1] > 0.7 * h:
        #     continue

        dist_pred = float(np.linalg.norm(center - pred_center)) if pred_center is not None else None
        dist_anchor = float(np.linalg.norm(center - anchor_center)) if anchor_center is not None else None

        box = cv2.boxPoints(rect).astype(np.int32)

        score = 0.0

        if dist_pred is not None:
            score += 2.0 * math.exp(-(dist_pred * dist_pred) / (2.0 * (120.0**2)))

        if dist_anchor is not None and rally_age is not None:
            sigma = 180.0
            fade = max(0.0, 1.0 - (rally_age / 20.0))
            score += (4.0 * fade) * math.exp(-(dist_anchor * dist_anchor) / (2.0 * sigma * sigma))

        info = {
            "idx": ci,
            "area": area,
            "cx": float(cx),
            "cy": float(cy),
            "bw": float(bw),
            "bh": float(bh),
            "bh_bw": float(bh_bw),
            "extent": float(extent),
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

    if frame_idx is not None and (frame_idx >= 120 and frame_idx <= 150):
        print(f"\n[frame {frame_idx}] rally contours={len(contours)} candidates={len(candidates)}")
        if pred_center is not None and prev_center is not None and prev_vel is not None:
            print(
                f"  pred=({pred_center[0]:.1f},{pred_center[1]:.1f}) "
                f"prev=({prev_center[0]:.1f},{prev_center[1]:.1f}) "
                f"v=({prev_vel[0]:.1f},{prev_vel[1]:.1f})"
            )
        if anchor_center is not None and rally_age is not None:
            print(f"  serve_anchor=({anchor_center[0]:.1f},{anchor_center[1]:.1f}) rally_age={rally_age}")

        top = sorted(candidates, key=lambda d: d["score"], reverse=True)[:10]
        print("  (top 10 by score)")
        for d in top:
            dist_pred_str = "" if d["dist_pred"] is None else f" dist_pred={d['dist_pred']:.1f}"
            dist_anch_str = "" if d["dist_anchor"] is None else f" dist_anchor={d['dist_anchor']:.1f}"
            print(
                f"  cand#{d['idx']:03d}: score={d['score']:.3f} "
                f"area={d['area']:.1f} center=({d['cx']:.1f},{d['cy']:.1f}) "
                f"bbox=({d['bw']:.0f}x{d['bh']:.0f}) extent={d['extent']:.2f} h/w={d['bh_bw']:.2f}"
                f"{dist_pred_str}{dist_anch_str}"
            )

        if best_center is None:
            print(f"[frame {frame_idx}] NO pick (no candidates)\n")
        else:
            print(
                f"[frame {frame_idx}] PICKED idx={best_info['idx']} center=({best_info['cx']:.1f},{best_info['cy']:.1f}) "
                f"area={best_info['area']:.1f} score={best_score:.3f}\n"
            )

    return best_center, best_box, candidates


def identify_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    fgsThresh = []
    cts = []

    prev_center = None
    prev_vel = np.array([0.0, 0.0], dtype=np.float32)

    serve_anchor = None
    rally_age = None

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        bw = cv2.medianBlur(bw, 3)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=1)

        thresh = bw
        fgsThresh.append(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ct_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(ct_img, contours, -1, (0, 255, 0), 2)

        if frame_idx < 120:
            center, radius, candidates = identify_serve(
                contours,
                frame_shape=ct_img.shape,
                prev_center=prev_center,
                frame_idx=frame_idx,
            )

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

                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                cv2.circle(ct_img, center_int, int(radius), (0, 255, 255), 2)

        else:
            rally_age = 0 if rally_age is None else rally_age + 1

            center, box, candidates = identify_rally(
                contours,
                frame_shape=ct_img.shape,
                prev_center=prev_center,
                prev_vel=prev_vel,
                frame_idx=frame_idx,
                serve_anchor=serve_anchor,
                rally_age=rally_age,
            )

            for d in candidates:
                cv2.drawContours(ct_img, [d["box"]], -1, (255, 0, 0), 2)

            if center is None and prev_center is not None:
                center = prev_center + prev_vel

            if center is not None:
                center_int = (int(center[0]), int(center[1]))
                if prev_center is not None:
                    prev_vel = center - prev_center
                prev_center = center

                cv2.circle(ct_img, center_int, 3, (0, 0, 255), -1)
                if box is not None:
                    cv2.drawContours(ct_img, [box], -1, (0, 255, 255), 2)

        cts.append(ct_img)
        frame_idx += 1

    font = cv2.FONT_HERSHEY_SIMPLEX

    video_th_wr = cv2.VideoWriter(
        "./footage/_extractedFGthresh.avi",
        cv2.VideoWriter.fourcc(*"MJPG"),
        50.0,
        frame_size,
        isColor=False,
    )
    for idx, fgT in enumerate(fgsThresh):
        fgT_annot = fgT.copy()
        cv2.putText(fgT_annot, f"Frame {idx}", (30, 100), font, 1.0, 255, 2, cv2.LINE_AA)
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
        cv2.putText(ct_annot, f"Frame {idx}", (30, 100), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        video_ct_wr.write(ct_annot)
    video_ct_wr.release()


# identify_ball("./footage/_cropped.avi")
identify_ball("./footage/_thresholded.avi")
# identify_ball("./footage/_thresholdedFG.avi")
