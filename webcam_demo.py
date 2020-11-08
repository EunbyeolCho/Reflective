import tensorflow as tf
import cv2
import time
import os
import argparse
import math
from dtaidistance import dtw
from posenet import *
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from playsound import playsound
from posenet.utils import read_cap

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 720)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

## 되는데 왜되는지 모르겠어서 빡친다
def counting_rightarm(keypoint_coords, old_raiseup):
    Count = False
    raiseup = old_raiseup
    if keypoint_coords[0, :, :][10][0] < keypoint_coords[0, :, :][12][0] + 20:
        ready = "ing"
        shoulder_min = keypoint_coords[0, :, :][6][0] - 15
        shoulder_max = keypoint_coords[0, :, :][6][0] + 15

        if shoulder_min < keypoint_coords[0, :, :][10][0] < shoulder_max:
            raiseup = True
    hip_min = keypoint_coords[0, :, :][12][0] - 15
    hip_max = keypoint_coords[0, :, :][12][0] + 15
    if old_raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        Count = True
        raiseup = False
    return Count, raiseup

## 일단은 팔을 올렸다가 다시 내렸을 때 음성 출력하도록 코딩
def checking_rightarm(keypoint_coords, old_raiseup, old_rightarm):
    Check = False
    raiseup = old_raiseup
    max_rightarm = min(old_rightarm, keypoint_coords[0, :, :][10][0])
    hip = keypoint_coords[0, :, :][12][0]
    shoulder_min = keypoint_coords[0, :, :][6][0] + 30
    shoulder_max = keypoint_coords[0, :, :][6][0] + 15
    if keypoint_coords[0, :, :][10][0] < hip + 20:
        if shoulder_max < max_rightarm < shoulder_min:
            raiseup = True
    hip_min = hip - 15
    hip_max = hip + 15
    if raiseup == True and hip_min < keypoint_coords[0, :, :][10][0] < hip_max:
        if shoulder_max< max_rightarm < shoulder_min:
            Check = True
            raiseup = False
            max_rightarm = 1000
    return Check, raiseup, max_rightarm

def imgkey():
    pos_temp_data = []
    sum = 0
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        f = r"D:\posenet-python-master\images\WIN_20201027_21_21_11_Pro.jpg"

        start = time.time()
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)
        # print("original img",input_image.shape)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.001)

        keypoint_coords *= output_scale
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                pos_temp_data.append(c[1])
                pos_temp_data.append(c[0])
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                pos_temp_data.append(s)
                sum = sum + s
            pos_temp_data.append(sum)

        draw_image = posenet.draw_skel_and_kp(
            draw_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.25, min_part_score=0.001)

        cv2.imshow("image key!",draw_image)
    return pos_temp_data, keypoint_coords

def compare(self, ip, model, i, j):
    ip = self.normalize(ip)
    scores = []
    for k in range(0, 17):
        scores.append(self.dtwdis(ip[:, k], model[:, k], i, j))
    return np.mean(scores), scores

def check_sim(correct_key, our_key):
    return dtw.distance(correct_key, our_key)

def cosine_similarity_our(a, b):
    #return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

def keypoint_compare(img_key, our_key, xmin, ymin, xmin_img, ymin_img):
    sum = 0
    for i in range(17):
        print(img_key[i]-[xmin_img, ymin_img], our_key[i]-[xmin,ymin])
        s = cosine_similarity(img_key[i]-[xmin_img, ymin_img], our_key[i]-[xmin,ymin])
        sum += s
    sum = sum/17
    return sum

def keypoint_compare_sklearn(img_key, our_key):
    sum = 0
    for i in range(17):
        # print(img_key[i], our_key[i])
        if img_key[i][0] == 0:
            z = np.zeros((400,400), dtype=np.uint8)
        # s = cosine_similarity(img_key[i], our_key[i])
        s = abs(cosine_similarity(img_key[i].reshape(1,-1), our_key[i].reshape(1,-1)))
        s = abs(eudist(s))
        sum += s
    sum = sum/17
    return sum

def normalize_sklearn(img_key, our_key):
    img_normalize_vec = []
    our_normalize_vec = []
    # print(our_key)
    # zero_our = np.where(our_key ==0)
    # print(zero_our)
    findzero = []

    for i in range(0,17):
        img_normalize_vec.append(img_key[i][0])
        img_normalize_vec.append(img_key[i][1])
        our_normalize_vec.append(our_key[i][0])
        our_normalize_vec.append(our_key[i][1])
    X = np.asarray([img_normalize_vec, our_normalize_vec], dtype=np.float)
    X_normalized = preprocessing.normalize(X, norm='l2')

    X_normalized_img = X_normalized[0,:].reshape(17,2, order='C')
    X_normalized_our = X_normalized[1,:].reshape(17,2, order='C')
    return X_normalized_img, X_normalized_our

def eudist(cosine):
    dist = 2*(1- cosine)
    dist = math.sqrt(dist)
    return dist

def main():
    img_key, img_key_coord = imgkey()
    ankle_height = 0
    counting = 0
    old_raiseup = False
    okay = False
    raiseup = False
    old_minwrist = 720
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        checkraiseup, rightarm = 0, 720
        if args.file is not None:
            cap = cv2.VideoCapture(r"D:\posenet-python-master\images\KakaoTalk_20201101_151405907.mp4")
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        framenum =0
        while True:
            framenum +=1
            pos_temp_data = []
            sum = 0

            input_image, display_image, output_scale = read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            print("video capture",input_image.shape)

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

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    pos_temp_data.append(c[1])
                    pos_temp_data.append(c[0])
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    pos_temp_data.append(s)
                    sum = sum + s
                pos_temp_data.append(sum)
            # print(keypoint_coords[0,:,:], "\n\n")
            sim = str(check_sim(pos_temp_data, img_key))

            # print("****")
            # print(keypoint_coords[0,:,:])
            coord = keypoint_coords[0,:,:]

            xmin = coord[:, 0].min()
            ymin = coord[:, 1].min()

            coord_img = img_key_coord[0,:,:]
            xmin_img = coord_img[:, 0].min()
            ymin_img = coord_img[:, 1].min()
            # print(img_key_coord[0,:,:])
            # print(keypoint_coords[0,:,:][4][1], keypoint_coords[0,:,:][2][1])
            # print(keypoint_coords[0,:,:])
            norm_img, norm_our = normalize_sklearn(img_key_coord[0,:,:], keypoint_coords[0,:,:])

            # if keypoint_coords[0,:,:][10][0] > keypoint_coords[0,:,:][12][0] + 20:
            #     ready = "True"
            # else:
            #     ready = "ing"
            #     shoulder_min = keypoint_coords[0, :, :][6][0] - 15
            #     shoulder_max = keypoint_coords[0, :, :][6][0] + 15
            #
            #     if shoulder_min < keypoint_coords[0,:,:][10][0] < shoulder_max:
            #         raiseup = True
            # hip_min = keypoint_coords[0, :, :][12][0] - 15
            # hip_max = keypoint_coords[0, :, :][12][0] + 15
            # if raiseup == True and hip_min<keypoint_coords[0, :, :][10][0]<hip_max:
            #     counting +=1
            #     raiseup = False
            #
            # old_raiseup = raiseup
            # cossim = keypoint_compare(img_key_coord[0,:,:], keypoint_coords[0,:,:], xmin, ymin,xmin_img, ymin_img)
            # rightarmcheck = check_rightarm(keypoint_coords, raiseup)
            Count, raiseup = counting_rightarm(keypoint_coords, raiseup)

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

            # Check, checkraiseup, rightarm = checking_rightarm(keypoint_coords, checkraiseup, rightarm)
            # if Check:
            #      playsound("./sound/rightarm.mp3")

            cossim = keypoint_compare_sklearn(norm_img, norm_our)
            dist = eudist(cossim)

            if cossim < 0.08:
                say ="True"
            else:
                say = "False"
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.1, min_part_score=0.1)

            cv2.putText(overlay_image, str(cossim), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),1)
            cv2.putText(overlay_image, str(counting), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
            cv2.imshow('posenet', overlay_image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()