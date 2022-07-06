import argparse
from pathlib import Path
import numpy as np
import time
import torch
from lib.model.Pose3DNet import Pose3DNet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image

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


def main():
    args = parse_args()

    # set cuda
    cuda = torch.device("cuda:%d" % args.gpu_id)
    pose3d_net = Pose3DNet(args).to(device=cuda)

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

    for i in range(2, len(all_img_paths)):
        since = time.time()
        imgs.append(load_image(all_img_paths[i]))
        imgs = imgs[-3:]

        img_tensor = torch.cat(imgs, dim=0)
        img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device=cuda)
        hm_tensors, offset_tensors = pose3d_net.forward(img_tensor)
        print("process time: ", time.time() - since)

        hm_tensors_np = hm_tensors.detach().cpu().numpy()
        offset_tensors_np = offset_tensors.detach().cpu().numpy()
        kps = calc_pose3d(hm_tensors_np[0], offset_tensors_np[0])

        args.out_dir.mkdir(exist_ok=True, parents=True)
        plot_pose3d(f"{args.out_dir}/{i:04d}.jpg", kps)


if __name__ == "__main__":
    main()
