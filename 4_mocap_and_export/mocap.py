import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
FOLDER_NAME = "images"
IMAGE_FILES = os.listdir(f'{FOLDER_NAME}/')
IMAGE_FILES.sort()
import json

def GetCenterXYH(landmarks):
    id_0 = mp_pose.PoseLandmark.RIGHT_HIP
    id_1 = mp_pose.PoseLandmark.LEFT_HIP
    id_2 = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    id_3 = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
    id_4 = mp_pose.PoseLandmark.NOSE

    return (
        -(landmarks[id_0].z + landmarks[id_1].z)/2,
        (landmarks[id_0].x + landmarks[id_1].x)/2,
        abs(landmarks[id_4].y - (landmarks[id_2].y + landmarks[id_3].y)/2)
    )

def GetTgtDir(landmarks, bone_name):
    is_single_bone = True

    # if bone_name == "shoulder_l":
    #     id_head = mp_pose.PoseLandmark.RIGHT_SHOULDER
    #     id_tail = mp_pose.PoseLandmark.LEFT_SHOULDER
    if bone_name == "upper_arm_l":
        id_head = mp_pose.PoseLandmark.LEFT_SHOULDER
        id_tail = mp_pose.PoseLandmark.LEFT_ELBOW
    elif bone_name == "lower_arm_l":
        id_head = mp_pose.PoseLandmark.LEFT_ELBOW
        id_tail = mp_pose.PoseLandmark.LEFT_WRIST
    # elif bone_name == "shoulder_r":
    #     id_head = mp_pose.PoseLandmark.LEFT_SHOULDER
    #     id_tail = mp_pose.PoseLandmark.RIGHT_SHOULDER
    elif bone_name == "upper_arm_r":
        id_head = mp_pose.PoseLandmark.RIGHT_SHOULDER
        id_tail = mp_pose.PoseLandmark.RIGHT_ELBOW
    elif bone_name == "lower_arm_r":
        id_head = mp_pose.PoseLandmark.RIGHT_ELBOW
        id_tail = mp_pose.PoseLandmark.RIGHT_WRIST
    # elif bone_name == "hip_l":
    #     id_head = mp_pose.PoseLandmark.RIGHT_HIP
    #     id_tail = mp_pose.PoseLandmark.LEFT_HIP
    elif bone_name == "thigh_l":
        id_head = mp_pose.PoseLandmark.LEFT_HIP
        id_tail = mp_pose.PoseLandmark.LEFT_KNEE
    elif bone_name == "shin_l":
        id_head = mp_pose.PoseLandmark.LEFT_KNEE
        id_tail = mp_pose.PoseLandmark.LEFT_ANKLE
    elif bone_name == "foot_l":
        id_head = mp_pose.PoseLandmark.LEFT_HEEL
        id_tail = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
    # elif bone_name == "hip_r":
    #     id_head = mp_pose.PoseLandmark.LEFT_HIP
    #     id_tail = mp_pose.PoseLandmark.RIGHT_HIP
    elif bone_name == "thigh_r":
        id_head = mp_pose.PoseLandmark.RIGHT_HIP
        id_tail = mp_pose.PoseLandmark.RIGHT_KNEE
    elif bone_name == "shin_r":
        id_head = mp_pose.PoseLandmark.RIGHT_KNEE
        id_tail = mp_pose.PoseLandmark.RIGHT_ANKLE
    elif bone_name == "foot_r":
        id_head = mp_pose.PoseLandmark.RIGHT_HEEL
        id_tail = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    else:
        is_single_bone = False

    if is_single_bone:
        return (
            - landmarks[id_tail].z + landmarks[id_head].z, 
            landmarks[id_tail].x - landmarks[id_head].x, 
            - landmarks[id_tail].y + landmarks[id_head].y, 
        )

    implemented = True

    if bone_name == "shoulder_l":
        tail_location = (
            -landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z, 
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
            -landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "shoulder_r":
        tail_location = (
            -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z, 
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
            -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "hip_l":
        tail_location = (
            -landmarks[mp_pose.PoseLandmark.LEFT_HIP].z, 
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
            -landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "hip_r":
        tail_location = (
            -landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z, 
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
            -landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "chest_l":
        tail_location = (
            -(landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z)/2, 
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)/2, 
            -(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)/2, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "chest_r":
        tail_location = (
            -(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)/2, 
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)/2, 
            -(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "hand_l":
        tail_location = (
            -(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].z + landmarks[mp_pose.PoseLandmark.LEFT_INDEX].z) / 2.0, 
            (landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x + landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x) / 2.0, 
            -(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y + landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y) / 2.0, 
        )
        head_location = (
            -landmarks[mp_pose.PoseLandmark.LEFT_WRIST].z, 
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
            -landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y, 
        )
    elif bone_name == "hand_r":
        tail_location = (
            -(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].z + landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].z) / 2.0, 
            (landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x + landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x) / 2.0, 
            -(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y + landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y) / 2.0, 
        )
        head_location = (
            -landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z, 
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
            -landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y, 
        )
    elif bone_name == "spine":
        tail_location = (
            -(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2.0, 
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2.0, 
            -(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2.0, 
        )
        head_location = (
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 4.0, 
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 4.0, 
            -(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 4.0, 
        )
    elif bone_name == "head":
        tail_location = (
            -landmarks[mp_pose.PoseLandmark.NOSE].z, 
            landmarks[mp_pose.PoseLandmark.NOSE].x,
            -landmarks[mp_pose.PoseLandmark.NOSE].y,
        )
        head_location = (
            -(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2.0, 
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2.0, 
            -(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2.0, 
        )
    else:
        implemented = False
        
    if implemented:
        return (
            tail_location[0] - head_location[0], 
            tail_location[1] - head_location[1], 
            tail_location[2] - head_location[2], 
        )

    if bone_name == "head_direction":
        head_direction = np.zeros((3))
        head_direction_inited = False
        if min(landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility) > 0.5:
            id_head = mp_pose.PoseLandmark.LEFT_EAR
            id_tail = mp_pose.PoseLandmark.RIGHT_EAR
            head_direction += np.array((
                - landmarks[id_tail].z + landmarks[id_head].z, 
                landmarks[id_tail].x - landmarks[id_head].x, 
                - landmarks[id_tail].y + landmarks[id_head].y, 
            ))
            head_direction_inited = True
        if min(landmarks[mp_pose.PoseLandmark.LEFT_EYE].visibility, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].visibility) > 0.5:
            id_head = mp_pose.PoseLandmark.LEFT_EYE
            id_tail = mp_pose.PoseLandmark.RIGHT_EYE
            head_direction += np.array((
                - landmarks[id_tail].z + landmarks[id_head].z, 
                landmarks[id_tail].x - landmarks[id_head].x, 
                - landmarks[id_tail].y + landmarks[id_head].y, 
            ))
            head_direction_inited = True
        if not head_direction_inited:
            print("head_direction low confidence")
            id_head = mp_pose.PoseLandmark.LEFT_SHOULDER
            id_tail = mp_pose.PoseLandmark.RIGHT_SHOULDER
            head_direction = np.array((
                - landmarks[id_tail].z + landmarks[id_head].z, 
                landmarks[id_tail].x - landmarks[id_head].x, 
                - landmarks[id_tail].y + landmarks[id_head].y, 
            ))
        return (head_direction[0], head_direction[1], head_direction[2])
    else:
        assert(False)

blender_data = []

pose_bone_name_list = [
    "spine", "head", "shoulder_l", "upper_arm_l", "lower_arm_l", "hand_l", "shoulder_r", "upper_arm_r", "lower_arm_r", "hand_r", "hip_l", "thigh_l",
    "shin_l", "foot_l", "hip_r", "thigh_r", "shin_r", "foot_r", "chest_l", "chest_r", "head_direction"
]

BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

  for file_name in tqdm(IMAGE_FILES[300:]):
    image = cv2.imread(f'{FOLDER_NAME}/' + file_name)

    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      print("not results.pose_landmarks:")
      break

    landmarks = results.pose_world_landmarks.landmark

    blender_data_entry = {}
    for pose_bone_name in pose_bone_name_list:
      blender_data_entry[pose_bone_name] = GetTgtDir(landmarks, pose_bone_name)
    blender_data_entry["center_xyh"] = GetCenterXYH(landmarks)

    blender_data.append(blender_data_entry)

with open("blender_data.json", "w") as blender_data_file:
    json.dump(blender_data, blender_data_file)