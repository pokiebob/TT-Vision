import cv2
import numpy as np
import math
from thresholding import erode_dilate, output_video
from tqdm import tqdm


def identify_ball2(video_path, min_area=50):
    # ***SETUP / LOAD VIDEO SECTION***
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

    # ***PRECOMPUTE CONTOURS SECTION***
    masked_grays = []
    contours_by_frame = erode_dilate("./footage/_thresholded.avi", min_area=min_area)

    prev_lines = []
    prev_ball = (-1, -1, [])  # (idx, angle, line)

    # ***MOTION / TRACK STATE SECTION***
    prev_ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    search_mode = 0
    SEARCH_FRAMES = 3

    FAR_JUMP_THRESH = 160
    TRACK_GATE = 100

    last_track_center = None
    last_track_vel = np.array([0.0, 0.0], dtype=np.float32)
    last_track_frame = None

    server_side = "?"
    server_locked = False
    STABLE_TRACK_FRAMES = 8
    stable_track_count = 0

    VEL_HIST_N = 12
    MAX_SPEED_ABS = 40.0
    SPEED_MED_MULT = 3.5
    ACC_ABS_THRESH = 25.0
    ACC_MED_MULT = 4.0
    SAME_DIR_DOT_MIN = 0.0

    track_speed_hist = []
    track_acc_hist = []

    # ***MAIN FRAME LOOP SECTION***
    for i, contours in enumerate(contours_by_frame):
        # ***RECT FITTING / CANDIDATE EXTRACTION SECTION***
        min_rects = [cv2.minAreaRect(ct) for ct in contours]
        min_boxes = (
            np.int64(np.array([cv2.boxPoints(rect) for rect in min_rects]))
            if len(min_rects)
            else np.zeros((0, 4, 2), dtype=np.int64)
        )
        boxes_filt = {"pass": [], "fail": []}

        contour_edge_frame = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
        passing = {"centers": [], "angles": [], "lines": [], "boxes": []}

        # ***PREDICTION (PREV CENTER + VELOCITY) SECTION***
        prev_ball_center = None
        pred_center = None
        if prev_ball[0] != -1:
            prev_ball_center = np.array(
                [
                    0.5 * (prev_ball[2][0][0] + prev_ball[2][1][0]),
                    0.5 * (prev_ball[2][0][1] + prev_ball[2][1][1]),
                ],
                dtype=np.float32,
            )
            pred_center = prev_ball_center + prev_ball_vel

        # ***OVER-TABLE HEURISTIC SECTION***
        over_table = False
        x_ref = None
        if pred_center is not None:
            x_ref = float(pred_center[0])
        elif prev_ball_center is not None:
            x_ref = float(prev_ball_center[0])
        if x_ref is not None:
            over_table = (0.3 * w <= x_ref <= 0.7 * w)

        hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)

        for j, box in enumerate(min_boxes):
            cx, cy = min_rects[j][0]
            boxW, boxH = min_rects[j][1]

            angle = np.deg2rad(min_rects[j][2])
            if boxW < boxH:
                angle += np.pi / 2

            l = 100
            pt1 = (round(cx - l * np.cos(angle)), round(cy - l * np.sin(angle)))
            pt2 = (round(cx + l * np.cos(angle)), round(cy + l * np.sin(angle)))

            [vx, vy, x0, y0] = cv2.fitLine(np.array([pt1, pt2]), cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy / (vx + 1e-6)
            intercept = y0 - slope * x0
            resid = np.ravel(contours[j][..., 1]) - (slope * np.ravel(contours[j][..., 0]) + intercept)
            mse = np.mean(resid ** 2)

            mask = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
            cv2.drawContours(mask, [box], -1, 255, cv2.FILLED)
            meanS = cv2.mean(hsv[..., 1], mask=mask)[0]

            # filters
            if not (20 < cv2.contourArea(box) < 2000):
                boxes_filt["fail"].append(box)
            elif boxW > 20 and boxH > 20:
                boxes_filt["fail"].append(box)
            elif meanS > 100:
                boxes_filt["fail"].append(box)
            elif np.abs(slope) > 1.7:
                boxes_filt["fail"].append(box)
            elif mse > 40:
                boxes_filt["fail"].append(box)
            else:
                boxes_filt["pass"].append(box)
                cv2.circle(frames[i], (int(cx), int(cy)), 3, (0, 0, 255), -1)

                cv2.line(frames[i], pt1, pt2, (120, 255, 120), 1)
                passing["centers"].append(np.array([cx, cy], dtype=np.float32))
                passing["angles"].append(angle)
                passing["lines"].append([pt1, pt2])
                passing["boxes"].append(box)

        print(f"--- Frame {i}: {len(passing['angles'])} passing ---")

        # ***HELPERS SECTION***
        def filter_passing(passing_dict, indices):
            new_passing = {"centers": [], "angles": [], "lines": [], "boxes": []}
            for idx in indices:
                for field in ["centers", "angles", "lines", "boxes"]:
                    new_passing[field].append(passing_dict[field][idx])
            return new_passing

        ball = (-1, -1, [])  # (idx, angle, line)

        # ***SCORING / SELECT BEST CANDIDATE SECTION***
        if len(passing["angles"]) > 0:
            if search_mode > 0:
                search_mode -= 1

            scored_idxs = []
            scores = []
            d_pred_list = []

            for j, cand_center in enumerate(passing["centers"]):
                cx, cy = cand_center

                d_pred = None
                pred_score = 0.0
                if pred_center is not None:
                    d_pred = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - pred_center))

                    if (search_mode == 0) and (d_pred > TRACK_GATE):
                        continue

                    sigma = 80.0 if over_table else 120.0
                    pred_score = math.exp(-(d_pred * d_pred) / (2.0 * sigma * sigma))

                center_score = 1.0 / (abs((cx / w) - 0.5) + 1e-6)

                if search_mode > 0:
                    score = 1.2 * center_score + 0.4 * pred_score
                elif over_table and pred_center is not None:
                    score = 0.4 * center_score + 4.0 * pred_score
                else:
                    score = 1.6 * center_score + 0.9 * pred_score

                scored_idxs.append(j)
                scores.append(score)
                d_pred_list.append(d_pred)

            if len(scores) > 0:
                k = int(np.argmax(scores))
                best = scored_idxs[k]
                ball = (best, passing["angles"][best], passing["lines"][best])

                if (search_mode == 0) and (pred_center is not None) and (d_pred_list[k] is not None) and (d_pred_list[k] > FAR_JUMP_THRESH):
                    search_mode = SEARCH_FRAMES
            else:
                search_mode = SEARCH_FRAMES
                ball = (-1, -1, [])
        else:
            print("nothing left, moving on")

        # ***GATING AROUND PREDICTED CENTER SECTION***
        if pred_center is not None:
            gate = TRACK_GATE if (search_mode == 0) else FAR_JUMP_THRESH

            keep_idx = []
            for j, cand_center in enumerate(passing["centers"]):
                d = float(np.linalg.norm(np.array(cand_center, dtype=np.float32) - pred_center))
                if d <= gate:
                    keep_idx.append(j)
                else:
                    boxes_filt["fail"].append(passing["boxes"][j])

            passing = filter_passing(passing, keep_idx)

            if ball[0] != -1:
                if ball[0] in keep_idx:
                    ball = (keep_idx.index(ball[0]), ball[1], ball[2])
                else:
                    ball = (-1, -1, [])

        # ***MOTION-BASED TRACK REJECTION SECTION***
        if (ball[0] != -1) and (search_mode == 0) and (last_track_center is not None) and (last_track_frame is not None):
            bx, by = passing["centers"][ball[0]]
            sel_center = np.array([bx, by], dtype=np.float32)

            dt = max(1, i - last_track_frame)

            v_cur = (sel_center - last_track_center) / float(dt)
            speed_cur = float(np.linalg.norm(v_cur))

            v_prev = last_track_vel
            dv = v_cur - v_prev
            acc_mag = float(np.linalg.norm(dv))
            dot_same_dir = float(np.dot(dv, v_prev))

            if dt == 1 and float(np.linalg.norm(v_prev)) > 2.0 and len(track_speed_hist) >= 6 and len(track_acc_hist) >= 6:
                med_speed = float(np.median(track_speed_hist))
                med_acc = float(np.median(track_acc_hist))

                too_fast = (speed_cur > MAX_SPEED_ABS) and (speed_cur > SPEED_MED_MULT * med_speed)
                too_much_acc = (acc_mag > ACC_ABS_THRESH) and (acc_mag > ACC_MED_MULT * med_acc)
                implausible_same_dir_boost = (too_much_acc and dot_same_dir > 0.25 * float(np.linalg.norm(v_prev))**2)

                if too_fast or implausible_same_dir_boost:
                    print(
                        f"[frame {i:03d}] REJECT TRACK candidate due to motion: "
                        f"speed={speed_cur:.1f} (med {med_speed:.1f}) "
                        f"|dv|={acc_mag:.1f} (med {med_acc:.1f}) "
                        f"dv·v_prev={dot_same_dir:.1f} dt={dt}"
                    )
                    search_mode = SEARCH_FRAMES
                    ball = (-1, -1, [])

        # ***VISUALIZATION: MODE / SEARCH BOX SECTION***
        mode_str = "SEARCH" if search_mode > 0 else "TRACK"
        if search_mode > 0 and last_track_center is not None:
            gate = FAR_JUMP_THRESH
            cx, cy = int(last_track_center[0]), int(last_track_center[1])

            x1 = int(max(0, cx - gate))
            y1 = int(max(0, cy - gate))
            x2 = int(min(w - 1, cx + gate))
            y2 = int(min(h - 1, cy + gate))

            # light blue box + pink dot
            cv2.rectangle(frames[i], (x1, y1), (x2, y2), (255, 200, 100), 2)
            cv2.circle(frames[i], (cx, cy), 6, (203, 192, 255), -1)

        cv2.drawContours(frames[i], passing["boxes"], -1, (0, 255, 0), 2)
        cv2.drawContours(frames[i], boxes_filt["fail"], -1, (150, 150, 255), 1)
        if ball[0] != -1:
            cv2.drawContours(frames[i], [passing["boxes"][ball[0]]], -1, (0, 255, 255), 3)

        # ***HUD + PRINTS SECTION***
        if ball[0] != -1:
            bx, by = passing["centers"][ball[0]]
            sel_center = np.array([bx, by], dtype=np.float32)

            bw_box = passing["boxes"][ball[0]]
            area = float(cv2.contourArea(bw_box))
            rw, rh = cv2.minAreaRect(bw_box)[1]

            d_prev = None if last_track_center is None else float(np.linalg.norm(sel_center - last_track_center))
            d_pred = None if pred_center is None else float(np.linalg.norm(sel_center - pred_center))

            v = None
            dv = None
            speed = None
            dspeed = None
            dt = None

            if last_track_center is not None and last_track_frame is not None:
                dt = max(1, i - last_track_frame)
                v = (sel_center - last_track_center) / float(dt)
                speed = float(np.linalg.norm(v))
                dv = v - last_track_vel
                dspeed = float(np.linalg.norm(dv))

            hud = [
                f"BALL @ ({int(bx)}, {int(by)})  mode={mode_str}",
                f"area={area:.0f}  box=({rw:.1f}x{rh:.1f})  d_prev={'NA' if d_prev is None else f'{d_prev:.1f}'}  d_pred={'NA' if d_pred is None else f'{d_pred:.1f}'}",
                f"v={'NA' if v is None else f'({v[0]:.1f},{v[1]:.1f})'}  |v|={'NA' if speed is None else f'{speed:.1f}'}   "
                f"dv={'NA' if dv is None else f'({dv[0]:.1f},{dv[1]:.1f})'}  |dv|={'NA' if dspeed is None else f'{dspeed:.1f}'}   dt={'NA' if dt is None else dt}",
            ]

            y0 = 140
            for k, txt in enumerate(hud):
                cv2.putText(
                    frames[i], txt, (20, y0 + 26 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 0), 2, cv2.LINE_AA
                )

            print(
                f"[frame {i:03d}] {mode_str} sel=({bx:.1f},{by:.1f}) "
                f"area={area:.0f} box=({rw:.1f}x{rh:.1f}) "
                f"d_prev={'NA' if d_prev is None else f'{d_prev:.1f}'} "
                f"d_pred={'NA' if d_pred is None else f'{d_pred:.1f}'} "
                f"v={'NA' if v is None else f'({v[0]:.1f},{v[1]:.1f})'} "
                f"|v|={'NA' if speed is None else f'{speed:.1f}'} "
                f"dv={'NA' if dv is None else f'({dv[0]:.1f},{dv[1]:.1f})'} "
                f"|dv|={'NA' if dspeed is None else f'{dspeed:.1f}'} "
                f"dt={'NA' if dt is None else dt} "
                f"over_table={over_table}"
            )
        else:
            print(
                f"[frame {i:03d}] NO BALL | mode={mode_str} | over_table={over_table} | passing={len(passing['centers'])}"
            )

        # ***SERVER SIDE LOCK SECTION***
        if not server_locked:
            if (ball[0] != -1) and (mode_str == "TRACK") and over_table:
                stable_track_count += 1
            else:
                stable_track_count = 0

            if stable_track_count >= STABLE_TRACK_FRAMES and (ball[0] != -1):
                bx, by = passing["centers"][ball[0]]
                server_side = "LEFT" if bx < (w / 2.0) else "RIGHT"
                server_locked = True

        server_text = f"server: {server_side}"
        (tw, th), _ = cv2.getTextSize(server_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            frames[i],
            server_text,
            (w - tw - 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ***FRAME LABELS SECTION***
        cv2.putText(
            frames[i],
            f"Frame {i}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frames[i],
            f"Mode: {mode_str} ({search_mode})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        table_text = "OVER TABLE" if over_table else "OUTSIDE TABLE"
        (tw2, _), _ = cv2.getTextSize(table_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            frames[i],
            table_text,
            (w - tw2 - 20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ***PREV BALL / PREDICTION VELOCITY UPDATE SECTION***
        prev_ball = (ball[0], ball[1], ball[2])

        if ball[0] != -1:
            new_center = np.array(
                [
                    0.5 * (ball[2][0][0] + ball[2][1][0]),
                    0.5 * (ball[2][0][1] + ball[2][1][1]),
                ],
                dtype=np.float32,
            )

            if prev_ball_center is not None:
                prev_ball_vel = new_center - prev_ball_center
            else:
                prev_ball_vel = 0.8 * prev_ball_vel

            prev_ball_center = new_center
        else:
            prev_ball_vel = 0.90 * prev_ball_vel

        # ***TRACK STATE + HISTORY UPDATE SECTION***
        if (ball[0] != -1) and (search_mode == 0):
            sel_center = passing["centers"][ball[0]]

            if last_track_center is not None and last_track_frame is not None:
                dt = max(1, i - last_track_frame)
                v_cur = (sel_center - last_track_center) / float(dt)
                speed_cur = float(np.linalg.norm(v_cur))
                dv = v_cur - last_track_vel
                acc_mag = float(np.linalg.norm(dv))

                track_speed_hist.append(speed_cur)
                track_acc_hist.append(acc_mag)
                if len(track_speed_hist) > VEL_HIST_N:
                    track_speed_hist.pop(0)
                if len(track_acc_hist) > VEL_HIST_N:
                    track_acc_hist.pop(0)

                last_track_vel = v_cur
            else:
                last_track_vel = np.array([0.0, 0.0], dtype=np.float32)

            last_track_center = sel_center.copy()
            last_track_frame = i

        prev_lines = passing["lines"]
        masked_grays.append(contour_edge_frame)

    # ***OUTPUT VIDEOS SECTION***
    output_video(frames, "_erodeDilateRects", isColor=True)
    output_video(masked_grays, "_maskedGray", isColor=False)


identify_ball2("./footage/_cropped.avi")
