import argparse
from pathlib import Path
import numpy as np
import cv2
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_sub", type=int, default=4)
    parser.add_argument("--n_frame", type=int, default=12)
    parser.add_argument("--n_keypoints", type=int, default=24)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    (args.out_dir / "RENDER").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "PARAMS").mkdir(parents=True, exist_ok=True)

    for i_sub in range(args.n_sub):
        for i_frame in range(args.n_frame):
            img = (np.random.rand(448, 448, 3) * 255).astype(np.uint8)
            out_img_path = args.out_dir / f"RENDER/{i_sub}/{i_frame:05d}.jpg"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), img)

            out_param_path = args.out_dir / f"PARAMS/{i_sub}/{i_frame:05d}.txt"
            out_param_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_param_path, "w") as f:
                kps = [random.random() * 2 - 1 for _ in range(n_keypoints * 3)]
                for v in np.reshape(kps, (-1, 3)):
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")


if __name__ == "__main__":
    main()
