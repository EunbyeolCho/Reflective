import tensorflow as tf
import time
import argparse
import math
from numpy.linalg import norm
from posenet import *
from playsound import playsound
from posenet.utils import read_cap

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 720)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--threshold', type =int, default = 3)
args = parser.parse_args()

def cosine_sim(x,y):
    return np.dot(x,y)/(norm(x)* norm(y))

def checking_shoulder_pix(keypoint_coords):
    l_sh_x = keypoint_coords[0, :, :][5][1]
    r_sh_x = keypoint_coords[0, :, :][6][1]
    print("shoulder is :::::::  ", l_sh_x - r_sh_x)
    return l_sh_x - r_sh_x

def counting_rightarm(keypoint_coords, old_raiseup, sh):
    Count = False
    raiseup = old_raiseup
    if keypoint_coords[0, :, :][10][0] < keypoint_coords[0, :, :][12][0] + 20:

        ## 사람마다 범위값이 달라지게 하려면, keypoint_coords[0, :, :][6][0] 을 원하는 값으로 대체.
        shoulder_min = keypoint_coords[0, :, :][6][0] - int(sh/4)
        shoulder_max = keypoint_coords[0, :, :][6][0] + int(sh/4)

        if shoulder_min < keypoint_coords[0, :, :][10][0] < shoulder_max:
            raiseup = True
    hip_min = keypoint_coords[0, :, :][12][0] - 15
    hip_max = keypoint_coords[0, :, :][12][0] + 15
    if old_raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        Count = True
        raiseup = False
    return Count, raiseup

def check_align(keypoint_coords):
    l_sh = keypoint_coords[0, :, :][5]
    r_sh = keypoint_coords[0, :, :][6]

    l_elb = keypoint_coords[0, :, :][7]
    r_elb = keypoint_coords[0, :, :][8]

    l_wr = keypoint_coords[0, :, :][9]
    r_wr = keypoint_coords[0, :, :][10]

    l_under_el = l_wr - l_elb
    r_under_el = r_wr - r_elb

    l_arm = l_wr - l_sh
    r_arm = r_wr - r_sh

    l_cos = cosine_sim(l_under_el, l_arm)
    r_cos = cosine_sim(r_under_el, r_arm)

    return l_cos, r_cos

## 일단은 팔을 올렸다가 다시 내렸을 때 음성 출력하도록 코딩
def checking_rightarm(keypoint_coords, old_raiseup, old_rightarm, sh):
    Check = False
    raiseup = old_raiseup
    max_rightarm = min(old_rightarm, keypoint_coords[0, :, :][10][0])
    hip = keypoint_coords[0, :, :][12][0]
    #사람마다 범위값이 달라지게 하려면, keypoint_coords[0, :, :][6][0] 을 원하는 값으로 대체.
    shoulder_min = keypoint_coords[0, :, :][6][0] + 2*int(sh/4)
    shoulder_max = keypoint_coords[0, :, :][6][0] + 2*int(sh/4)
    if keypoint_coords[0, :, :][10][0] < hip + int(sh/4):
        if shoulder_max < max_rightarm < shoulder_min:
            raiseup = True
    hip_min = hip - int(sh/4)
    hip_max = hip + int(sh/4)
    if raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        if shoulder_max< max_rightarm < shoulder_min:
            Check = True
            raiseup = False
            max_rightarm = 1000
    return Check, raiseup, max_rightarm

def main():
    counting = 0
    old_raiseup = False
    raiseup = False
    align, notalign =0, 0
    old_minwrist = 720
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        print(model_outputs)

        output_stride = model_cfg['output_stride']
        checkraiseup, rightarm = 0, 720
        if args.file is not None:
            cap = cv2.VideoCapture(r"D:\posenet-python-master\images\ligt_oneleg_correct.mp4")
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, 1080)
        cap.set(4, 720)

        start = time.time()
        frame_count = 0
        framenum =0
        while True:
            framenum +=1
            pos_temp_data = []

            input_image, display_image, output_scale = read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            print("video capture",input_image.shape)
            print(input_image.shape)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            print(heatmaps_result.shape, offsets_result.shape, displacement_fwd_result.shape, displacement_bwd_result.shape)
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
            if math.acos(r_cos) > 0.15:
                notalign +=1
            elif math.acos(r_cos) < 0.10:
                align +=1

            if align > 30:
                align, notalign = 0, 0

            sh = checking_shoulder_pix(keypoint_coords)
            Count, raiseup = counting_rightarm(keypoint_coords, raiseup, sh)

            rightwrist = keypoint_coords[0, :, :][10][0]
            minwrist = min(rightwrist, old_minwrist)
            shoulder_min = keypoint_coords[0, :, :][6][0] + 30
            shoulder_max = keypoint_coords[0, :, :][6][0] + 15
            hip_min =  keypoint_coords[0, :, :][12][0] - 15
            hip_max =  keypoint_coords[0, :, :][12][0] + 15
            if Count:
                counting +=1
                minwrist = 720
            if shoulder_max<minwrist < shoulder_min and hip_min< rightwrist < hip_max:
                playsound("./sound/rightarm.mp3")
                minwrist = 720
            old_minwrist = minwrist

            Check, checkraiseup, rightarm = checking_rightarm(keypoint_coords, checkraiseup, rightarm, sh)
            #if Check:
                # playsound("./sound/rightarm.mp3")
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)
            if notalign >= 20:
                print("Align your ankle-knee-hip!!")
                cv2.putText(overlay_image, "Align Error", (150, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255),1 )
            cv2.putText(overlay_image, str(counting), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
            cv2.imshow('posenet', overlay_image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()