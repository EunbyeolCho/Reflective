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

from posenet.utils import read_cap

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 720)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

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
        print("original img",input_image.shape)
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
            min_pose_score=0.25)

        # keypoint_coords *= output_scale
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


    return pos_temp_data, keypoint_coords

def compare(self, ip, model, i, j):
    ip = self.normalize(ip)
    scores = []
    for k in range(0, 17):
        scores.append(self.dtwdis(ip[:, k], model[:, k], i, j))
    return np.mean(scores), scores

def check_sim(correct_key, our_key):
    return dtw.distance(correct_key, our_key)

def cosine_similarity(a, b):

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
        print(img_key[i], our_key[i])
        # s = cosine_similarity(img_key[i], our_key[i])
        s = cosine_similarity(img_key[i], our_key[i])
        sum += s
    sum = sum/17
    return sum

def normalize_sklearn(img_key, our_key):
    img_normalize_vec = []
    our_normalize_vec = []
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
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        global framenum
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
                min_pose_score=0.15)

            # keypoint_coords *= output_scale

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

            print("****")
            # print(keypoint_coords[0,:,:])
            coord = keypoint_coords[0,:,:]

            xmin = coord[:, 0].min()
            ymin = coord[:, 1].min()

            coord_img = img_key_coord[0,:,:]
            xmin_img = coord_img[:, 0].min()
            ymin_img = coord_img[:, 1].min()
            # print(img_key_coord[0,:,:])
            norm_img, norm_our = normalize_sklearn(img_key_coord[0,:,:], keypoint_coords[0,:,:])
            # cossim = keypoint_compare(img_key_coord[0,:,:], keypoint_coords[0,:,:], xmin, ymin,xmin_img, ymin_img)
            cossim = keypoint_compare_sklearn(norm_img, norm_our)
            dist = eudist(cossim)
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.putText(overlay_image, str(dist), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255),1)
            cv2.imshow('posenet', overlay_image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()