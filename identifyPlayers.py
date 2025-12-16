# identifyPlayers.py
import cv2
import numpy as np


def identify_players(contours, video_path,
                     x_left_max_ratio=0.30,
                     x_right_min_ratio=0.70,
                     min_area=220,
                     seed_top_k=6,
                     max_link_dist_ratio=0.18,
                     pad_ratio=0.055, 
                     y_min_ratio=0.22,
                     y_max_ratio=0.95,
                     min_cluster_contours=3,
                     min_cluster_area_ratio=0.010, 
                     max_box_aspect=2.6,
                     min_box_w_ratio=0.10,
                     min_box_h_ratio=0.14):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_area = float(w * h)

    x_left_max = x_left_max_ratio * w
    x_right_min = x_right_min_ratio * w

    BASE_LINK = max_link_dist_ratio * w
    PAD = pad_ratio * w

    y_min = y_min_ratio * h
    y_max = y_max_ratio * h

    def feat(c):
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + 0.5 * bw
        cy = y + 0.5 * bh
        area = float(cv2.contourArea(c))
        return (cx, cy, area, x, y, bw, bh)

    feats = [feat(c) for c in contours]

    feats = [f for f in feats if f[2] >= min_area and (y_min <= f[1] <= y_max)]

    left = [f for f in feats if f[0] <= x_left_max]
    right = [f for f in feats if f[0] >= x_right_min]

    def grow_cluster(side_feats):
        if not side_feats:
            return None

        # pick closest to median y
        side_sorted = sorted(side_feats, key=lambda t: t[2], reverse=True)
        topk = side_sorted[:min(seed_top_k, len(side_sorted))]
        med_y = float(np.median([t[1] for t in topk]))
        seed = min(topk, key=lambda t: abs(t[1] - med_y))

        selected = [seed]
        remaining = [t for t in side_sorted if t is not seed]

        # iterative growth with a tightening gate:
        for pass_idx in range(6):
            if not remaining:
                break

            cx = float(np.mean([t[0] for t in selected]))
            cy = float(np.mean([t[1] for t in selected]))

            size_factor = 1.0 / np.sqrt(max(1.0, float(len(selected))))
            gate = BASE_LINK * (0.85 ** pass_idx) * (0.9 + 0.6 * size_factor)
            gate = max(0.09 * w, gate)  # don't go below a small floor

            new_sel = []
            still_rem = []
            for t in remaining:
                d = float(np.hypot(t[0] - cx, t[1] - cy))
                if d <= gate:
                    new_sel.append(t)
                else:
                    still_rem.append(t)

            if not new_sel:
                break

            selected.extend(new_sel)
            remaining = still_rem

        cluster_area = float(sum(t[2] for t in selected))
        if (len(selected) < min_cluster_contours) and (cluster_area < min_cluster_area_ratio * frame_area):
            return None

        # union bbox
        xs1 = [t[3] for t in selected]
        ys1 = [t[4] for t in selected]
        xs2 = [t[3] + t[5] for t in selected]
        ys2 = [t[4] + t[6] for t in selected]

        x1 = int(max(0, min(xs1) - PAD))
        y1 = int(max(0, min(ys1) - PAD))
        x2 = int(min(w - 1, max(xs2) + PAD))
        y2 = int(min(h - 1, max(ys2) + PAD))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # reject boxes that are too small
        if bw < (min_box_w_ratio * w) or bh < (min_box_h_ratio * h):
            return None

        # reject extreme aspect ratios
        aspect = float(max(bw, bh)) / float(min(bw, bh) + 1e-6)
        if aspect > max_box_aspect:
            return None

        return (x1, y1, x2, y2)

    return grow_cluster(left), grow_cluster(right)
