import argparse
from pathlib import Path
from re import sub

import cv2
import numpy as np
import scipy.spatial.transform

target_bone_names = [
    "Hips",
    "Spine",
    "Neck",
    "Head", 
    # "Nose"
    #    "RightEye",
    #   "LeftEye",
    #    "RightEar",
    #   "LeftEar",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandMiddle1",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandMiddle1",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToe_End",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root_dir", type=Path, required=True)
    return parser.parse_args()

def create_translation_mat(x: float, y: float, z: float):
    trans_mat = np.eye(4)
    trans_mat[:, 3] = [x, y, z, 1]

    return trans_mat


def create_scale_mat(x: float, y: float, z: float):
    scale_mat = np.eye(4)

    scale_mat[0, 0] = x
    scale_mat[1, 1] = y
    scale_mat[2, 2] = z

    return scale_mat

def create_look_at_mat(eye, target, up=[0.0, 0.0, 1.0]):
    eye = np.array(eye, dtype=np.float32, copy=True)
    target = np.array(target, dtype=np.float32, copy=True)
    up = np.array(up, dtype=np.float32, copy=True)

    z = eye - target
    z = z / np.linalg.norm(z)

    x = np.cross(up, z)
    x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    rotate_mat = np.array([
        [x[0], x[1], x[2], 0],
        [y[0], y[1], y[2], 0],
        [z[0], z[1], z[2], 0],
        [0, 0, 0, 1]
    ])
    
    scale_mat = create_scale_mat(1, -1, -1)
    tmp = np.dot(scale_mat, rotate_mat)

    return tmp[:3, :3]

def load_param(param_path):
    mtx = np.eye(3)

    params = {}
    with param_path.open("r") as f:
        for line in f:
            line = line.replace("=", ",").replace(";", ",")
            tokens = line.split(",")
            if tokens[0] == "extrinsic":
                # parse 16 floats as 4x4 matrix
                extrinsic = np.array([float(t) for t in tokens[3 : 3 + 16]]).reshape(
                    4, 4
                )
            elif tokens[0] == "extrinsic_euler":
                extrinsic_euler = np.array([float(t) for t in tokens[3 : 3 + 3]])
            elif tokens[0] == "intrinsic":
                mtx = np.array([float(t) for t in tokens[3 : 3 + 9]]).reshape(3, 3)
                mtx[1, 1] = mtx[0, 0]
            else:
                R = create_look_at_mat(extrinsic[:3, 3], (0, 0, 1))
                p = extrinsic[:3, 3]
                p = -np.matmul(R, p.T).T
                pose = np.zeros((3, 4), float)
                pose[:3, :3] = R
                pose[:3, 3] = p.ravel()

                bone_name = tokens[0]
                if ':' in bone_name:
                    bone_name = bone_name.split(":")[1]

                head_tail = tokens[1]
                pt = np.array([float(t) for t in tokens[2 : 3 + 3]])
                for ti, target in enumerate(target_bone_names):
                    if target == bone_name and head_tail == "head":
                        params[target] = (pt, pose, mtx)

    # check if params is len(target_bone_names)
    assert len(params) == len(target_bone_names), str(param_path)

    return params


def convert_param(params):
    new_params = {}
    for target, (pt, pose, mtx) in params.items():
        kp = cv2.projectPoints(np.array([pt]), pose[:3, :3], pose[:3, 3], mtx, None)[0][
            0
        ].ravel().astype(int)
        new_pt = pose @ np.append(pt, 1).T
        new_params[target] = *new_pt, *kp

    return new_params

def visualize_keypoint(img, keypoints):
    for _, kp in keypoints.items():
        cv2.circle(img, (kp[3], kp[4]), 3, (0, 0, 255), -1)
    return img


def save_param(param_path, params):
    with param_path.open("w") as f:
        for target in target_bone_names:
            f.write(",".join([str(v) for v in params[target]]) + "\n")


def main(args):
    # glob and convert to list
    all_img_paths = list(args.root_dir.glob("RENDER/*/*.png"))
    
    bbox = np.array([np.inf, np.inf, np.inf]), np.array([-np.inf, -np.inf, -np.inf])

    latest_ng = None
    for i, img_path in enumerate(all_img_paths):
        if i % 1000 == 0:
            print("{}/{}".format(i, len(all_img_paths)))
        if latest_ng == img_path.parent:
            continue
        
        sub_name = img_path.parent.name
        param_path = args.root_dir / "PARAMS_RAW" / sub_name / (img_path.stem + ".txt")
        params = load_param(param_path)
        new_params = convert_param(params)
        
        n_ok = True
        for _, kp in new_params.items():
            if kp[3] < 0 or 448 <= kp[3] or kp[4] < 0 or 448 <= kp[4]:
                pass
            else:
                n_ok += 1
                
        # 二個ミスなら許容
        if n_ok >= len(target_bone_names) - 2:
            # out_param_path = args.root_dir / "PARAMS" / sub_name / (img_path.stem + ".txt")
            # out_param_path.parent.mkdir(parents=True, exist_ok=True)
            # save_param(out_param_path, new_params)
            
            for _, kp in new_params.items():
                bbox = np.minimum(bbox[0], kp[:3]), np.maximum(bbox[1], kp[:3])
        else:
            latest_ng = img_path.parent
            print(f"{str(img_path)} has out-of-range keypoints. Skip the remaining images.")
            
            render = visualize_keypoint(cv2.imread(str(img_path)), new_params)
            cv2.imshow("render", render)
            cv2.waitKey(0)
        

    print(bbox)


if __name__ == "__main__":
    args = parse_args()
    main(args)
