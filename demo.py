import argparse
from operator import is_
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import lib.visualize as visualize
from lib.model.Pose3DNet import K, Pose3DNet

DEFAULT_CKPT_PATH = r"D:\workspace\InstantPose3D\train_mini\ckpt\epoch_135.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--load_size", type=int, default=448)
    args = parser.parse_args()
    return args


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def plot_pose3d(save_path, kps):
    fig = plt.figure()
    ax = Axes3D(fig)
    # set limit ax
    ax.set_xlim3d(0, 30)
    ax.set_ylim3d(0, 30)
    ax.set_zlim3d(0, 30)

    xs, ys, zs = kps[:, 0], kps[:, 1], kps[:, 2]
    ax.scatter(xs, ys, zs, c="#ff0000", s=30, marker="o")

    # save figure
    plt.savefig(save_path)


def calc_pose3d(hm, offset):
    # max index of gt_hm_tensor_np
    kps = []
    for ki in range(offset.shape[0]):
        i, j, k = np.unravel_index(np.argmax(hm[ki]), hm[ki].shape)
        x = offset[ki, i, j, k, 0] + i + 0.5
        y = offset[ki, i, j, k, 1] + j + 0.5
        z = offset[ki, i, j, k, 2] + k + 0.5
        kps.append([x, y, z])
    return np.array(kps)


def anim_create():
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    return ax


def anim_begin_update(ax, world_size=0.5):
    ax.cla()
    ax.set_title("3D Points")
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_zlim(-world_size, world_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def anim_end_update(ax, interval=1e-3):
    fig = ax.figure
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(interval)


def show_plt(block=True):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show(block=block)


def plot_skeleton3d(ax, points3d, kps_colors_plt):
    # keypoints
    visualize.plot_points3d(ax, points3d, kps_colors_plt, center=(0, 0, 0), s=128)

    # bones
    # for a, b in skeleton:
    #     if np.any(points3d[a] != 0) and np.any(points3d[b] != 0):
    #         x1, y1, z1 = points3d[a]
    #         x2, y2, z2 = points3d[b]
    #         ax.plot(
    #             [x1, x2],
    #             [y1, y2],
    #             [z1, z2],
    #             "o-",
    #             c=kps_colors_plt[a],
    #             ms=0,
    #             mew=64,
    #         )


def main():
    args = parse_args()

    # set cuda
    is_onnx = False
    if args.pretrained_path.suffix == ".onnx":
        import onnxruntime as ort
        is_onnx = True
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "pose3d_opt.onnx"

        ort_session = ort.InferenceSession(str(args.pretrained_path), sess_options, providers=["CUDAExecutionProvider"])
    else:
        cuda = torch.device("cuda:%d" % args.gpu_id)
        pose3d_net = Pose3DNet(args, is_train=False).to(device=cuda)

        if args.pretrained_path is not None:
            print("=============")
            print("loading checkpoint model ...", args.pretrained_path)
            print("=============")
            state_dict = torch.load(args.pretrained_path, map_location=cuda)
            pose3d_net.load_state_dict(state_dict)
        pose3d_net.eval()

    # PIL to tensor
    to_tensor = transforms.Compose(
        [
            transforms.Resize(args.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    all_img_paths = list(args.input_dir.glob("*.jpg"))

    def load_image(img_path):
        img = Image.open(img_path).convert("RGB")
        return to_tensor(img)

    imgs = []
    for i in range(2):
        imgs.append(load_image(all_img_paths[i]))

    ax = visualize.create_plt(28)
    visualize.show_plt(False)

    kps_colors = [
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for _ in range(K)
    ]
    kps_colors_plt = [(r / 255, g / 255, b / 255) for r, g, b in kps_colors]

    for i in range(2, len(all_img_paths)):
        since = time.time()
        imgs.append(load_image(all_img_paths[i]))
        imgs = imgs[-3:]

        img_tensor = torch.cat(imgs, dim=0)
        img_tensor = img_tensor.unsqueeze(0)

        if is_onnx:
            hm_tensors_np, offset_tensors_np = ort_session.run(
                None,
                {"input": img_tensor.detach().cpu().numpy() },
            )
        else:
            img_tensor = img_tensor.to(device=cuda)
            hm_tensors, offset_tensors = pose3d_net.forward(img_tensor)
            hm_tensors_np = hm_tensors.detach().cpu().numpy()
            offset_tensors_np = offset_tensors.detach().cpu().numpy()
        print("process time: ", time.time() - since)

        kps = calc_pose3d(hm_tensors_np[0], offset_tensors_np[0])

        # Draw 3D
        visualize.anim_begin_update(ax, world_size=28)
        plot_skeleton3d(ax, kps, kps_colors_plt)
        visualize.anim_end_update(ax)


#        args.out_dir.mkdir(exist_ok=True, parents=True)
#       plot_pose3d(f"{args.out_dir}/{i:04d}.jpg", kps)


if __name__ == "__main__":
    main()
