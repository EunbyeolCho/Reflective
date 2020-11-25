import tensorflow as tf
import cv2
import time
import argparse
import math
from dtaidistance import dtw
from posenet import *
from posenet.utils import read_cap
import numpy as np
from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 720)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def cosine_sim(x,y):
    return np.dot(x,y)/(norm(x)* norm(y))
#keypoint_coords[0, :, :][joint num][0]
def kneepoint_check(keypoint_scores, min_part_score=0.3):
    r_knee_score = keypoint_scores[0, :][11]
    l_knee_score = keypoint_scores[0, :][12]

    if r_knee_score < min_part_score or l_knee_score < min_part_score:
        return True
    return False

def equilibrium(keypoint_coords):
    count = 0
    shoulder_diff = keypoint_coords[0, :, :][6][0] - keypoint_coords[0, :, :][5][0]
    hip_diff = keypoint_coords[0, :, :][12][0] -  keypoint_coords[0, :, :][11][0]
    ankle_diff = keypoint_coords[0, :, :][16][0] -  keypoint_coords[0, :, :][15][0]

    hip_x_diff = keypoint_coords[0, :, :][12][1] -  keypoint_coords[0, :, :][11][1]
    ankle_x_diff = keypoint_coords[0, :, :][16][1] -  keypoint_coords[0, :, :][15][1]

    if abs(ankle_x_diff) < 20 or abs(hip_x_diff) < 20:
        return False
    if abs(shoulder_diff) > 10:
        count +=1
    if abs(hip_diff) > 10:
        count +=1
    if abs(ankle_diff) > 10:
        count +=1
    if count < 1:
        return True
    return False

def check_left_lunge(keypoint_coords):
    r_thigh_y = keypoint_coords[0, :, :][11][0]
    r_knee_y = keypoint_coords[0,:,:][13][0]
    r_ankle_y = keypoint_coords[0, :, :][15][0]
    # print(r_knee_y - r_thigh_y,2 * (r_ankle_y - r_knee_y) )
    if r_knee_y - r_thigh_y > 2 * (r_ankle_y - r_knee_y):
        return True
    return False

def check_align(keypoint_coords):
    l_hip = keypoint_coords[0, :, :][12]
    r_hip = keypoint_coords[0, :, :][11]

    l_knee = keypoint_coords[0, :, :][14]
    r_knee = keypoint_coords[0, :, :][13]

    l_ankle = keypoint_coords[0, :, :][16]
    r_ankle = keypoint_coords[0, :, :][15]

    l_calf_vec = l_ankle - l_knee
    r_calf_vec = r_ankle - r_knee

    l_leg_vec = l_ankle - l_hip
    r_leg_vec = r_ankle - r_hip

    l_cos = cosine_sim(l_calf_vec, l_leg_vec)
    r_cos = cosine_sim(r_calf_vec, r_leg_vec)

    return l_cos, r_cos

def main():
    counting = 0
    oldeq = True
    notalign = 0
    align = 0
    ready = False
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        if args.file is not None:
            cap = cv2.VideoCapture(r"D:\posenet-python-master\images\lunge.mp4")
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        framenum =0
        check = False
        while True:
            print("lets go")
            framenum +=1
            input_image, display_image, output_scale = read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.001)

            keypoint_coords *= output_scale
            l_cos , r_cos = check_align(keypoint_coords)

            # print(math.acos(l_cos))
            if math.acos(l_cos) > 0.15:
                notalign +=1
            elif math.acos(l_cos) < 0.10:
                align +=1

            if align > 30:
                align, notalign = 0, 0

            if kneepoint_check(keypoint_scores):
                continue

            eq = equilibrium(keypoint_coords)
            if oldeq == True and eq == False:
                ready = True
            if ready is True:
                if check_left_lunge(keypoint_coords):
                    check = True

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)
            # cv2.imshow("i'm overlay", overlay_image)
            if notalign >= 20:
                print("Align your ankle-knee-hip!!")
                ready, eq, check = False, False, False
                cv2.putText(overlay_image, "Align Error", (150, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255),1 )

            if oldeq == False and eq == True and check == True:
                counting +=1
                ready = False
                check = False
            oldeq = eq
            cv2.putText(overlay_image, str(counting), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
            cv2.imshow('posenet', overlay_image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()