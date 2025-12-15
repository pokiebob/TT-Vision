import cv2
import numpy as np
import math
from thresholding import erode_dilate, output_video
from tqdm import tqdm


def identify_ball2(video_path, min_area=50):
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

    prev_ball_center = None
    prev_ball_vel = np.array([0.0, 0.0], dtype=np.float32)

    search_mode = 0
    SEARCH_FRAMES = 3

    FAR_JUMP_THRESH = 160
    TRACK_GATE = 100




    for i, contours in enumerate(contours_by_frame):
        min_rects = [cv2.minAreaRect(ct) for ct in contours]
        min_boxes = np.int64(np.array([cv2.boxPoints(rect) for rect in min_rects]))
        boxes_filt = {"pass": [], "fail": []}

        contour_edge_frame = np.zeros_like(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))
        passing = {"centers": [], "angles": [], "lines": [], "boxes": []}


        prev_ball_center = None
        pred_center = None

        if prev_ball[0] != -1:
            center_point = np.array([
                0.5*(prev_ball[2][0][0] + prev_ball[2][1][0]),
                0.5*(prev_ball[2][0][1] + prev_ball[2][1][1]),
            ], dtype=np.float32)

            prev_ball_center = center_point
            pred_center = prev_ball_center + prev_ball_vel


        over_table = False
        x_ref = None
        if pred_center is not None:
            x_ref = pred_center[0]
        elif prev_ball_center is not None:
            x_ref = prev_ball_center[0]

        if x_ref is not None:
            over_table = (0.3*w <= x_ref <= 0.7*w)



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
            if search_mode > 0:
                search_mode -= 1

            scored_idxs = []
            scores = []
            d_pred_list = []

            for j, cand_center in enumerate(passing["centers"]):
                cx, cy = cand_center

                # distance to predicted
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



        # if len(passing["angles"]) > 0:
        #     center_pos = []
        #     for j, cand in enumerate(passing["centers"]):
        #         center_pos.append(passing["centers"][j][0]/w)
        #     if len(center_pos) > 0:
        #         center_pos = np.array(center_pos)
        #         center_scores = 1 / np.abs(center_pos - 0.5)
        #         closest_center = np.argmax(center_scores)
        #         if np.max(center_scores) >= 5:
        #             print("setting ball to closest center (score > 5)")
        #             ball = (closest_center, passing["angles"][closest_center], passing["lines"][closest_center])
        #             prev_centered = True
        #         else:
        #             print("no ball found in center of frame")
        #             #prev_centered = False

        #     if ball[0] == -1 and prev_ball[0] != -1:  # if no ball in center, but we have a previous ball
        #         # find closest to previous ball center, similar slope
        #         dist_to_prev = [np.linalg.norm(np.array([0.5*(cand[0][0] + cand[1][0]), 0.5*(cand[0][1] + cand[1][1])]) - center_point) for cand in passing["lines"]]
        #         min_dist = np.argmin(dist_to_prev)
        #         #if np.abs(passing["angles"][min_dist] - prev_ball[1]) < np.pi/2:
        #         #    print("setting ball to closest to prev ball with slope within pi/2")
        #         #    ball = (min_dist, passing["angles"][min_dist], passing["lines"][min_dist])
        #         ball = (min_dist, passing["angles"][min_dist], passing["lines"][min_dist])
        else:
            print("nothing left, moving on")

        masked_grays.append(contour_edge_frame)

        prev_lines = passing["lines"]

        if pred_center is not None:
            gate = TRACK_GATE if (search_mode == 0) else FAR_JUMP_THRESH 

            keep_idx = []
            for j, cand_center in enumerate(passing["centers"]):
                cx, cy = cand_center
                d = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - pred_center))

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


        cv2.drawContours(frames[i], passing["boxes"], -1, (0,255,0), 2)
        cv2.drawContours(frames[i], boxes_filt["fail"], -1, (150,150,255), 1)
        if ball[0] != -1:
            cv2.drawContours(frames[i], [passing["boxes"][ball[0]]], -1, (0,255,255), 3)
        
        prev_ball = (ball[0], ball[1], ball[2])
        
        #update velocity
        if ball[0] != -1:
            new_center = np.array([
                0.5*(ball[2][0][0] + ball[2][1][0]),
                0.5*(ball[2][0][1] + ball[2][1][1]),
            ], dtype=np.float32)

            if prev_ball_center is not None:
                prev_ball_vel = new_center - prev_ball_center
            else:
                prev_ball_vel = 0.8 * prev_ball_vel

            prev_ball_center = new_center
        else:
            
            prev_ball_vel = 0.90 * prev_ball_vel


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
            f"Mode: {'SEARCH' if search_mode > 0 else 'TRACK'} ({search_mode})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        table_text = "OVER TABLE" if over_table else "OUTSIDE TABLE"

        (tw, th), _ = cv2.getTextSize(
            table_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )

        x_right = w - tw - 20

        cv2.putText(
            frames[i],
            table_text,
            (x_right, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )



    output_video(frames, "_erodeDilateRects", isColor=True)
    output_video(masked_grays, "_maskedGray", isColor=False)
    

identify_ball2("./footage/_cropped.avi")