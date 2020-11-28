import tensorflow as tf
import cv2
import time
import argparse

import posenet
import posenet.constants

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=r'C:\Users\samsung\Pictures\video\weight_shift_crop.mp4', help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def find_shld_minmax(min_x, max_x, start_keypoints, keypoints):
    start_right_shoulder_x = start_keypoints[0][6][1]
    right_shoulder_x = keypoints[0][6][1]
    if right_shoulder_x < start_right_shoulder_x:
        if right_shoulder_x < min_x:
            min_x = right_shoulder_x
    if right_shoulder_x > start_right_shoulder_x:
        if right_shoulder_x > max_x:
            max_x = right_shoulder_x
    return min_x, max_x

def check_count(min_x, max_x, start_keypoints, keypoints):
    start_right_shoulder_x = start_keypoints[0][6][1]
    right_shoulder_x = keypoints[0][6][1]
    if int(max_x - min_x) >= 80 and right_shoulder_x <= start_right_shoulder_x:
        return True
    else:
        return False

def detect_wrong(wrong_count, keypoints):
    right_shoulder_y = keypoints[0][6][0]
    left_shoulder_y = keypoints[0][5][0]
    if abs(right_shoulder_y - left_shoulder_y) > 30:
        wrong_count += 1
    return wrong_count

def check_align(keypoints):
    right_shoulder_x = keypoints[0][6][1]
    left_shoulder_x = keypoints[0][5][1]
    right_waist_x = keypoints[0][12][1]
    left_waist_x = keypoints[0][11][1]

    mid_shoulder_x = right_shoulder_x + (left_shoulder_x - right_shoulder_x)/2
    mid_waist_x = right_waist_x + (left_waist_x - right_waist_x)/2

    if mid_waist_x-10 < mid_shoulder_x < mid_waist_x+10:
        return True
    else:
        return False


def main():
    start = time.time()
    frame_count = 0
    is_align = False
    align = 0
    start_exercise = False
    min_shld_x = args.cam_width
    max_shld_x = 0
    counting = 0
    increase_count = False
    wrong_counting = 0

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if 'png' in args.file:
            img = cv2.imread(args.file) #r'C:\Users\samsung\Pictures\demo\weightshift-wrong.png'
        else:
            # cap = cv2.VideoCapture(args.file) #r'C:\Users\samsung\Pictures\video\체중이동.mp4'
            cap = cv2.VideoCapture(args.cam_id)
            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)
            # web_cap.set(3, args.cam_width)
            # web_cap.set(4, args.cam_height)

        while True:

            if 'png' in args.file:
                input_image, display_image, output_scale = posenet.utils._process_input(img, 1.0, 16)
            else:
                # input_image: preprocessed / display_image: source image
                input_image, display_image, output_scale = posenet.read_cap(
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
                max_pose_detections=5,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            color = (0, 255, 255)

            # check align to decide start
            if not is_align or align < 32:
                is_align = check_align(keypoint_coords)
                if is_align == True:
                    align += 1
                if align == 32: #1초동안 align하면
                    start_exercise = True
                    cv2.putText(overlay_image, "Start", (50, 50), font, scale, color)
                    start_keypoints = keypoint_coords

            if start_exercise == True and frame_count % 5 == 0:
                # count
                min_shld_x, max_shld_x = find_shld_minmax(min_shld_x, max_shld_x, start_keypoints, keypoint_coords)
                increase_count = check_count(min_shld_x, max_shld_x, start_keypoints, keypoint_coords)
                if increase_count:
                    counting += 1
                    print("==============count + 1 ===============")
                    min_shld_x = args.cam_width
                    max_shld_x = 0

                # wrong 자세 판단
                wrong_counting = detect_wrong(wrong_counting, keypoint_coords)
                if wrong_counting >= 2:
                    # print("Right_y: ", right_shoulder_y, " Left_y: ", left_shoulder_y)
                    print("!!!!!!!!! Wrong !!!!!!!!!")
                    wrong_counting = 0
                    wrong_scale = 3
                    cv2.putText(overlay_image, "wrong", (100, 300), font, wrong_scale, color, 3)


            cv2.putText(overlay_image, str(counting), (50, 50), font, scale, color, 3)
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()