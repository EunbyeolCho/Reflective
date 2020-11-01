import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

#Set up
# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }


Bodyparts = ["Head", "Neck", "Rshoulder", "RElbow", 
                "Rwrist", "LShoulder", "LElbow", "LWrist",
                "RHip", "RKnee", "RAnkle", "LHip", 
                "LKnee", "LAnkle", "Chest", "Background"]

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]


protoFile="./models/pose/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./models/pose/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def GetCosineSimilarity(A, B):

    a0, a1 = A[0] + 0.001 , A[1] + 0.001
    b0, b1 = B[0] + 0.001, B[1] + 0.001 

    a = tuple([a0, a1])
    b = tuple([b0, b1])

    csim = dot(a, b)/(norm(a)*norm(b))
    if csim > 1 : csim = 1

    return csim




def demo(image_dir):

    image = cv2.imread(image_dir)
    image = cv2.resize(image, dsize=(350, 450))

    imageHeight, imageWidth, _ = image.shape
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)
    output = net.forward()
    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비

    H = output.shape[2]
    W = output.shape[3]
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

    #Draw keypoints on images
    points = []
    for i in range(0,15):

        #Confidence map
        probMap = output[0, i, :, :]
    
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.1 :     
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # cv2.imshow("Output-Keypoints",image)
    # cv2.waitKey(0)

    return points, image



def RemoveNoneIdx(input_points, gt_points) :

    if None in input_points :
        none_list = [i for i, input_point in enumerate(input_points) if input_point == None]

        for none_idx in none_list : 

            if input_points[none_idx-1]:
                input_points[none_idx] = input_points[none_idx-1]
                gt_points[none_idx] =gt_points[none_idx-1]
            else : 
                input_points[none_idx] = input_points[none_idx+1]
                gt_points[none_idx] =gt_points[none_idx+1]

    if None in gt_points:
        none_list = [i for i, gt_point in enumerate(gt_points) if gt_points == None]

        for none_idx in none_list : 

            if gt_points[none_idx-1]:
                input_points[none_idx] = input_points[none_idx-1]
                gt_points[none_idx] =gt_points[none_idx-1]
            else : 
                input_points[none_idx] = input_points[none_idx+1]
                gt_points[none_idx] =gt_points[none_idx+1]

    return input_points, gt_points



if __name__ == "__main__":

    input_img_dir = "./sample_images/a.jpg"
    gt_img_dir = "./sample_images/b.jpg"

    input_points, input_image = demo(input_img_dir)
    gt_points, gt_image = demo(gt_img_dir)
    # print("Input Points : ", input_points)
    # print("GT Points : ", gt_points)
    
    input_points, gt_points = RemoveNoneIdx(input_points, gt_points)

    csim_sum = 0
    for i in range(15):  
        csim = abs(GetCosineSimilarity(gt_points[i], input_points[i]))
        csim_sum += csim
        print(Bodyparts[i]," ====> ", csim)
    
    print(csim_sum / 15) #0.98974292

    # numpy_vertical = np.hstack((input_image, gt_image))
    # cv2.imshow("Output-Keypoints",numpy_vertical)
    # cv2.waitKey(0)


    #####################################
    
    count = 0
    csim_sum = 0
    input_image = cv2.resize(cv2.imread(input_img_dir), dsize=(350, 450))
    gt_image = cv2.resize(cv2.imread(gt_img_dir), dsize=(350, 450))

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        print(pair, end=" : ")
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
        
        #print(partA," 와 ", partB, " 연결\n")
        if input_points[partA] and input_points[partB]:
            cv2.line(input_image, input_points[partA], input_points[partB], (0, 255, 0), 2)
            a = np.array(input_points[partB]) - np.array(input_points[partA])
            
        if gt_points[partA] and gt_points[partB]:
            cv2.line(gt_image, gt_points[partA], gt_points[partB], (0, 0, 255), 2)
            b = np.array(gt_points[partB]) - np.array(gt_points[partA])
        
        
        csim = GetCosineSimilarity(a, b)
        print(csim, " ===> Degree : ", math.degrees(math.acos(csim)))
        csim_sum += csim
        count +=1
                
    print("Final score : ", csim_sum / count) #0.98974292

    


    points = []
    for i in range(0,15):

        input_x, input_y = input_points[i]
        gt_x, gt_y = gt_points[i]

        cv2.circle(input_image, (int(input_x), int(input_y)), 3, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(gt_image, (int(gt_x), int(gt_y)), 3, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)

    numpy_vertical = np.hstack((input_image, gt_image))
    cv2.imshow("Output-Keypoints",numpy_vertical)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


