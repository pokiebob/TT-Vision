import cv2
import numpy as np
import math
from thresholding import erode_dilate, output_video
# from tqdm import tqdm
from identifyPlayers import identify_players


# ***BALL IDENTIFICATION MAIN FUNCTION SECTION***
def identify_ball2(video_path, min_area=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    masked_grays = []
    contours_by_frame = erode_dilate("./footage/_thresholded.avi", min_area=min_area)

    prev_ball = (-1, -1, [])  # (idx, angle, line)

    # ***MOTION STATE (GENERAL) SECTION***
    prev_ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    # ***SEARCH/TRACK STATE SECTION***
    search_mode = 0
    SEARCH_FRAMES = 3

    FAR_JUMP_THRESH = 140
    TRACK_GATE = 120

    last_track_center = None
    last_track_vel = np.array([0.0, 0.0], dtype=np.float32)
    last_track_frame = None

    last_any_center = None
    last_any_vel = np.array([0.0, 0.0], dtype=np.float32)
    last_any_frame = None

    BASE_SEARCH_GATE = FAR_JUMP_THRESH
    SEARCH_VEL_GAIN = 1.8
    MAX_SEARCH_GATE = int(0.45 * max(w, h))

    TABLE_ANCHOR = np.array([0.5 * w, 0.55 * h], dtype=np.float32)
    TABLE_Y_MIN = 0.30 * h
    TABLE_Y_MAX = 0.75 * h

    TRUST_UPDATE_D_PRED = 220.0




    # ***SERVER SIDE LOCK STATE SECTION***
    server_side = "?"
    server_locked = False

    STABLE_TRACK_FRAMES = 10
    stable_track_count = 0

    # ***MOTION REJECTION THRESHOLDS SECTION***
    VEL_HIST_N = 12
    MAX_SPEED_ABS = 100.0
    SPEED_MED_MULT = 3.5
    ACC_ABS_THRESH = 25.0
    ACC_MED_MULT = 4.0

    track_speed_hist = []
    track_acc_hist = []

    # ***PRE-LOCK INITIAL SEARCH SECTION***
    prelock_mode = True

    # ***HIT-AWARE REACQUIRE SECTION***
    HIT_COS_THRESH = -0.3
    HIT_SPEED_ABS = 80.0

    def cos_sim(a, b):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 1.0
        return float(np.dot(a, b) / (na * nb))
    
    # ***BALL STATE SECTION***
    ball_state = "air"
    ball_dir_str = "->"

    HIT_BOX_PAD = 18

    def point_in_box(p, box):
        if box is None:
            return False
        x, y = float(p[0]), float(p[1])
        x1, y1, x2, y2 = box
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def pad_box(box, pad, w, h):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return (
            int(max(0, x1 - pad)),
            int(max(0, y1 - pad)),
            int(min(w - 1, x2 + pad)),
            int(min(h - 1, y2 + pad)),
        )

    def vel_dir_horiz(v):
        vx = float(v[0])
        if abs(vx) < 1e-6:
            return None
        return "->" if vx > 0 else "<-"

    def dir_from_side(side, server_side):
        if side == "LEFT":
            return "->"
        if side == "RIGHT":
            return "<-"
        if server_side == "LEFT":
            return "->"
        if server_side == "RIGHT":
            return "<-"
        return "->"





    # ***FRAME LOOP SECTION***
    for i, contours in enumerate(contours_by_frame):
        prev_search_mode = search_mode

        min_rects = [cv2.minAreaRect(ct) for ct in contours]
        min_boxes = (
            np.int64(np.array([cv2.boxPoints(rect) for rect in min_rects]))
            if len(min_rects)
            else np.zeros((0, 4, 2), dtype=np.int64)
        )
        boxes_filt = {"pass": [], "fail": []}

        contour_edge_frame = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
        passing = {"centers": [], "angles": [], "lines": [], "boxes": []}

        # ***PREDICTION CENTER COMPUTATION SECTION***
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

        # ***CANDIDATE GENERATION + GEOMETRY FILTERS SECTION***
        for j, box in enumerate(min_boxes):
            cx, cy = min_rects[j][0]
            boxW, boxH = min_rects[j][1]

            angle = np.deg2rad(min_rects[j][2])
            if boxW < boxH:
                angle += np.pi / 2

            l = 100
            pt1 = (round(cx - l * np.cos(angle)), round(cy - l * np.sin(angle)))
            pt2 = (round(cx + l * np.cos(angle)), round(cy + l * np.sin(angle)))

            [vx, vy, x0, y0] = cv2.fitLine(
                np.array([pt1, pt2]), cv2.DIST_L2, 0, 0.01, 0.01
            )
            slope = vy / (vx + 1e-6)
            intercept = y0 - slope * x0
            resid = np.ravel(contours[j][..., 1]) - (
                slope * np.ravel(contours[j][..., 0]) + intercept
            )
            mse = np.mean(resid**2)

            mask = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
            cv2.drawContours(mask, [box], -1, 255, cv2.FILLED)
            meanS = cv2.mean(hsv[..., 1], mask=mask)[0]

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

        # ***HELPER: FILTER PASSING DICT SECTION***
        def filter_passing(passing_dict, indices):
            new_passing = {"centers": [], "angles": [], "lines": [], "boxes": []}
            for idx in indices:
                for field in ["centers", "angles", "lines", "boxes"]:
                    new_passing[field].append(passing_dict[field][idx])
            return new_passing

        ball = (-1, -1, [])  # (idx, angle, line)

        # ***PRELOCK MODE (NO TRACK/SEARCH GATING) SECTION***
        if prelock_mode:
            if len(passing["centers"]) > 0:
                scores = []
                for (cx, cy) in passing["centers"]:
                    center_score = 1.0 / (abs((cx / w) - 0.5) + 1e-6)
                    scores.append(center_score)
                best = int(np.argmax(scores))
                ball = (best, passing["angles"][best], passing["lines"][best])
            else:
                ball = (-1, -1, [])

            sel_over_table = False
            if ball[0] != -1:
                bx, by = passing["centers"][ball[0]]
                sel_over_table = (0.3 * w <= float(bx) <= 0.7 * w)

            if ball[0] != -1 and sel_over_table:
                stable_track_count += 1
            else:
                stable_track_count = 0

            if stable_track_count >= STABLE_TRACK_FRAMES and ball[0] != -1:
                bx, by = passing["centers"][ball[0]]
                server_side = "LEFT" if bx < (w / 2.0) else "RIGHT"
                server_locked = True
                prelock_mode = False

                last_track_center = np.array([bx, by], dtype=np.float32)
                last_track_frame = i
                last_track_vel = np.array([0.0, 0.0], dtype=np.float32)
                prev_ball_vel = np.array([0.0, 0.0], dtype=np.float32)

                ball_dir_str = "->" if server_side == "LEFT" else "<-"
                ball_state = "air"

        # ***NORMAL SEARCH/TRACK SECTION (ONLY AFTER PRELOCK)***
        if not prelock_mode:
            search_anchor = None
            if pred_center is not None:
                search_anchor = pred_center
            elif last_track_center is not None:
                search_anchor = last_track_center

            if (
                search_mode > 0
                and search_anchor is not None
                and len(passing["centers"]) > 0
            ):
                v_ref = prev_ball_vel
                if last_any_vel is not None and last_any_frame is not None and last_any_frame < i:
                    v_ref = last_any_vel

                speed_ref = float(np.linalg.norm(v_ref))
                gate = int(min(BASE_SEARCH_GATE + SEARCH_VEL_GAIN * speed_ref, MAX_SEARCH_GATE))

                keep = []
                for j, c in enumerate(passing["centers"]):
                    d = float(np.linalg.norm(np.array(c, dtype=np.float32) - search_anchor))
                    if d <= gate:
                        keep.append(j)
                    else:
                        boxes_filt["fail"].append(passing["boxes"][j])
                passing = filter_passing(passing, keep)

            ball = (-1, -1, [])
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

                    anchor = (
                        search_anchor
                        if (search_mode > 0 and search_anchor is not None)
                        else pred_center
                    )
                    if anchor is not None:
                        d_pred = float(
                            np.linalg.norm(np.array([cx, cy], dtype=np.float32) - anchor)
                        )

                        if (
                            (search_mode == 0)
                            and (pred_center is not None)
                            and (d_pred > TRACK_GATE)
                        ):
                            continue

                        sigma = 80.0 if over_table else 120.0
                        pred_score = math.exp(-(d_pred * d_pred) / (2.0 * sigma * sigma))

                    center_score = 1.0 / (abs((cx / w) - 0.5) + 1e-6)

                    if search_mode > 0:
                        table_pen = 0.0
                        if cy < TABLE_Y_MIN or cy > TABLE_Y_MAX:
                            table_pen = 1.0

                        table_score = math.exp(
                            -((float(cy) - float(TABLE_ANCHOR[1])) ** 2) / (2.0 * (0.18 * h) ** 2)
                        )

                        score = 2.2 * center_score + 2.6 * pred_score + 1.6 * table_score - 3.0 * table_pen

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

                    if (
                        (search_mode == 0)
                        and (pred_center is not None)
                        and (d_pred_list[k] is not None)
                        and (d_pred_list[k] > FAR_JUMP_THRESH)
                    ):
                        search_mode = SEARCH_FRAMES
                else:
                    search_mode = SEARCH_FRAMES
                    ball = (-1, -1, [])
            else:
                search_mode = SEARCH_FRAMES

            if (
                (ball[0] != -1)
                and (search_mode == 0)
                and (last_track_center is not None)
                and (last_track_frame is not None)
            ):
                bx, by = passing["centers"][ball[0]]
                sel_center = np.array([bx, by], dtype=np.float32)

                ref_center = last_track_center
                ref_vel = last_track_vel
                ref_frame = last_track_frame

                if (
                    prev_search_mode > 0
                    and last_any_center is not None
                    and last_any_frame is not None
                    and last_any_frame < i
                ):
                    ref_center = last_any_center
                    ref_vel = last_any_vel
                    ref_frame = last_any_frame

                dt = max(1, i - ref_frame)
                v_cur = (sel_center - ref_center) / float(dt)
                speed_cur = float(np.linalg.norm(v_cur))

                dv = v_cur - ref_vel
                acc_mag = float(np.linalg.norm(dv))

                med_speed = float(np.median(track_speed_hist)) if len(track_speed_hist) >= 5 else 0.0
                med_acc = float(np.median(track_acc_hist)) if len(track_acc_hist) >= 5 else 0.0

                too_fast = (speed_cur > MAX_SPEED_ABS) or (
                    med_speed > 1e-6 and speed_cur > SPEED_MED_MULT * med_speed
                )
                too_much_acc = (acc_mag > ACC_ABS_THRESH) or (
                    med_acc > 1e-6 and acc_mag > ACC_MED_MULT * med_acc
                )

                just_reacquired = (prev_search_mode > 0 and search_mode == 0)
                flip = cos_sim(v_cur, last_track_vel) < HIT_COS_THRESH
                plausible_hit = speed_cur <= HIT_SPEED_ABS

                if just_reacquired and flip and plausible_hit:
                    last_track_center = sel_center
                    last_track_frame = i
                    last_track_vel = v_cur
                    track_speed_hist = []
                    track_acc_hist = []
                elif too_fast or too_much_acc:
                    print(
                        f"[frame {i:03d}] REJECT TRACK candidate due to motion: "
                        f"speed={speed_cur:.1f} (med {med_speed:.1f}) "
                        f"|dv|={acc_mag:.1f} (med {med_acc:.1f}) dt={dt}"
                    )
                    search_mode = SEARCH_FRAMES
                    ball = (-1, -1, [])

            mode_str = "SEARCH" if search_mode > 0 else "TRACK"
            if search_mode > 0 and search_anchor is not None:
                v_ref = prev_ball_vel
                if last_any_vel is not None and last_any_frame is not None and last_any_frame < i:
                    v_ref = last_any_vel
                speed_ref = float(np.linalg.norm(v_ref))
                gate = int(min(BASE_SEARCH_GATE + SEARCH_VEL_GAIN * speed_ref, MAX_SEARCH_GATE))

                cx, cy = int(search_anchor[0]), int(search_anchor[1])
                x1 = int(max(0, cx - gate))
                y1 = int(max(0, cy - gate))
                x2 = int(min(w - 1, cx + gate))
                y2 = int(min(h - 1, cy + gate))
                cv2.rectangle(frames[i], (x1, y1), (x2, y2), (255, 200, 100), 2)
                cv2.circle(frames[i], (cx, cy), 6, (203, 192, 255), -1)
        else:
            mode_str = "PRELOCK"

        # ***DRAW DEBUG CONTOURS SECTION***
        cv2.drawContours(frames[i], passing["boxes"], -1, (0, 255, 0), 2)
        cv2.drawContours(frames[i], boxes_filt["fail"], -1, (150, 150, 255), 1)
        if ball[0] != -1:
            cv2.drawContours(
                frames[i], [passing["boxes"][ball[0]]], -1, (0, 255, 255), 3
            )
        left_box, right_box = identify_players(contours, video_path)

        purple = (255, 0, 255)
        if left_box is not None:
            x1, y1, x2, y2 = left_box
            cv2.rectangle(frames[i], (x1, y1), (x2, y2), purple, 4)

        if right_box is not None:
            x1, y1, x2, y2 = right_box
            cv2.rectangle(frames[i], (x1, y1), (x2, y2), purple, 4)

        if (ball[0] != -1) and (not prelock_mode) and server_locked:
            bx, by = passing["centers"][ball[0]]
            ball_center = np.array([bx, by], dtype=np.float32)

            left_box_p = pad_box(left_box, HIT_BOX_PAD, w, h)
            right_box_p = pad_box(right_box, HIT_BOX_PAD, w, h)

            in_left = point_in_box(ball_center, left_box_p)
            in_right = point_in_box(ball_center, right_box_p)

            if in_left:
                ball_state = "hit"
                ball_dir_str = "->"
            elif in_right:
                ball_state = "hit"
                ball_dir_str = "<-"
            else:
                ball_state = "air"

        print(
            f"[frame {i:03d}] server_side={server_side} locked={server_locked} prelock={prelock_mode} "
            f"state={ball_state} dir={ball_dir_str}"
        )

        # ***HUD / LOGGING SECTION***
        if ball[0] != -1:
            bx, by = passing["centers"][ball[0]]
            bw_box = passing["boxes"][ball[0]]
            area = float(cv2.contourArea(bw_box))
            rw, rh = cv2.minAreaRect(bw_box)[1]
            print(
                f"[frame {i:03d}] {mode_str} sel=({bx:.1f},{by:.1f}) area={area:.0f} box=({rw:.1f}x{rh:.1f}) over_table={over_table}"
            )
        else:
            print(
                f"[frame {i:03d}] NO BALL | mode={mode_str} | over_table={over_table} | passing={len(passing['centers'])}"
            )

        # ***STATIC LABELS / TEXT SECTION***
        server_text = f"Server: {server_side}"
        (tw, th), _ = cv2.getTextSize(
            server_text, cv2.FONT_HERSHEY_SIMPLEX, float(1.0), int(2)
        )
        cv2.putText(
            frames[i],
            server_text,
            (w - tw - 20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(1.0),
            (255, 255, 255),
            int(2),
            cv2.LINE_AA,
        )

        cv2.putText(
            frames[i],
            f"Frame {i}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(1.0),
            (255, 255, 255),
            int(2),
            cv2.LINE_AA,
        )
        cv2.putText(
            frames[i],
            f"Mode: {mode_str} ({search_mode})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(0.9),
            (255, 255, 255),
            int(2),
            cv2.LINE_AA,
        )
        cv2.putText(
            frames[i],
            f"Ball state: {ball_state}   dir: {ball_dir_str}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        

        dbg_lines = []
        dbg_lines.append(f"passing: {len(passing['centers'])}")

        if pred_center is not None:
            dbg_lines.append(f"pred: ({pred_center[0]:.1f},{pred_center[1]:.1f})")
        else:
            dbg_lines.append("pred: None")

        if last_track_center is not None:
            dbg_lines.append(f"last: ({last_track_center[0]:.1f},{last_track_center[1]:.1f})")
        else:
            dbg_lines.append("last: None")

        if ball[0] != -1:
            bx, by = passing["centers"][ball[0]]
            dbg_lines.append(f"sel: ({bx:.1f},{by:.1f})")

            if pred_center is not None:
                d_pred = float(np.linalg.norm(np.array([bx, by], dtype=np.float32) - pred_center))
                dbg_lines.append(f"d_pred: {d_pred:.1f}")
            else:
                dbg_lines.append("d_pred: None")

            if last_track_center is not None and last_track_frame is not None:
                dt = max(1, i - last_track_frame)
                v_est = (np.array([bx, by], dtype=np.float32) - last_track_center) / float(dt)
                spd = float(np.linalg.norm(v_est))
                dbg_lines.append(f"dt: {dt}  v: ({v_est[0]:.1f},{v_est[1]:.1f})  |v|: {spd:.1f}")
                dv = v_est - last_track_vel
                acc = float(np.linalg.norm(dv))
                dbg_lines.append(f"|dv|: {acc:.1f}")
            else:
                dbg_lines.append("dt: None  v: None")
        else:
            dbg_lines.append("sel: None")
            dbg_lines.append("d_pred: None")
            dbg_lines.append("dt: None  v: None")

        x_right = w - 20
        y0 = 100
        dy = 22
        for li, t in enumerate(dbg_lines):
            (lw, lh), _ = cv2.getTextSize(
                t, cv2.FONT_HERSHEY_SIMPLEX, float(0.6), int(2)
            )
            cv2.putText(
                frames[i],
                t,
                (x_right - lw, y0 + li * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(0.6),
                (255, 255, 255),
                int(2),
                cv2.LINE_AA,
            )

        if ball[0] != -1:
            bx, by = passing["centers"][ball[0]]
            cur_center_any = np.array([bx, by], dtype=np.float32)

            ok_y = (TABLE_Y_MIN <= float(by) <= TABLE_Y_MAX)

            ok_d = True
            if pred_center is not None:
                d_pred_any = float(np.linalg.norm(cur_center_any - pred_center))
                ok_d = (d_pred_any <= TRUST_UPDATE_D_PRED)

            if ok_y and ok_d:
                if last_any_center is not None and last_any_frame is not None:
                    dt_any = max(1, i - last_any_frame)
                    last_any_vel = (cur_center_any - last_any_center) / float(dt_any)
                else:
                    last_any_vel = 0.8 * last_any_vel

                last_any_center = cur_center_any
                last_any_frame = i



        # ***PREV BALL UPDATE SECTION***
        prev_center_for_vel = prev_ball_center
        prev_ball = (ball[0], ball[1], ball[2])

        # ***VELOCITY UPDATE FOR PREDICTION SECTION***
        if ball[0] != -1:
            new_center = np.array(
                [
                    0.5 * (ball[2][0][0] + ball[2][1][0]),
                    0.5 * (ball[2][0][1] + ball[2][1][1]),
                ],
                dtype=np.float32,
            )
            if prev_center_for_vel is not None:
                prev_ball_vel = new_center - prev_center_for_vel
            else:
                prev_ball_vel = 0.8 * prev_ball_vel
        else:
            prev_ball_vel = 0.90 * prev_ball_vel

        # ***TRACK STATE + HISTORIES UPDATE SECTION***
        if (ball[0] != -1) and (not prelock_mode) and (search_mode == 0):
            bx, by = passing["centers"][ball[0]]
            sel_center = np.array([bx, by], dtype=np.float32)

            if last_track_center is not None and last_track_frame is not None:
                dt = max(1, i - last_track_frame)
                v_cur = (sel_center - last_track_center) / float(dt)
                speed_cur = float(np.linalg.norm(v_cur))

                dv = v_cur - last_track_vel
                acc_mag = float(np.linalg.norm(dv))

                track_speed_hist.append(speed_cur)
                track_acc_hist.append(acc_mag)
                if len(track_speed_hist) > VEL_HIST_N:
                    track_speed_hist = track_speed_hist[-VEL_HIST_N:]
                if len(track_acc_hist) > VEL_HIST_N:
                    track_acc_hist = track_acc_hist[-VEL_HIST_N:]

                last_track_vel = v_cur
            else:
                last_track_vel = np.array([0.0, 0.0], dtype=np.float32)

            last_track_center = sel_center
            last_track_frame = i

        masked_grays.append(contour_edge_frame)

    # ***OUTPUT VIDEOS SECTION***
    output_video(frames, "_erodeDilateRects", isColor=True)
    # output_video(masked_grays, "_maskedGray", isColor=False)


if __name__ == "__main__":
    identify_ball2("./footage/_cropped.avi")
