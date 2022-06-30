import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from lib.model.Pose3DNet import Pose3DNet, hg_layer_nums, K
from lib.dataset.TrainDataset import TrainDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, default=None)
    parser.add_argument("--checkpoints_path", type=Path, required=True)

    # 学習パラメタ
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # データセット・モデル
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--output_size", type=int, default=228)

    # TODO: augment系

    parser.add_argument("--hg_depth", type=int, default=2)
    parser.add_argument("--frame_num", type=int, default=130)
    parser.add_argument("--load_size", type=int, default=448)

    args = parser.parse_args()
    return args


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def plot_heatmap(heatmap):
    fig = plt.figure()
    ax = Axes3D(fig)
    # set limit ax
    ax.set_xlim3d(0, 30)
    ax.set_ylim3d(0, 30)
    ax.set_zlim3d(0, 30)

    xs, ys, zs = [], [], []
    colors, alpha = [], []

    sizes = []
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for k in range(heatmap.shape[2]):
                if heatmap[i, j, k] < 0.02:
                    continue
                v = max(0, min(1, heatmap[i, j, k] * 30))
                xs.append(i)
                ys.append(j)
                zs.append(k)
                colors.append(rgb_to_hex((255 * v, 0, 255 * (1 - v))))
                alpha.append(v)
                sizes.append(np.exp(heatmap[i, j, k] * 200))

    ax.scatter(xs, ys, zs, c=colors, alpha=alpha, s=sizes, marker="o")
    plt.show()


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
    #    cuda = torch.device("cpu")

    # Setup network
    # https://www.youtube.com/watch?v=kKNcyTqybjA&ab_channel=%E9%9D%92%E6%9F%B3%E5%B9%B8%E5%BD%A6
    pose3d_net = Pose3DNet(args).to(device=cuda)

    if args.pretrained_path is not None:
        print("=============")
        print("loading pretrained model ...", args.pretrained_path)
        print("=============")
        state_dict = torch.load(args.pretrained_path, map_location=cuda)
        pose3d_net.load_state_dict(state_dict)

    # Create output checkpionts directory
    args.checkpoints_path.mkdir(exist_ok=True, parents=True)

    # 学習パラメタ: TODO
    hm_criterion = torch.nn.BCEWithLogitsLoss()
    of_criterion = torch.nn.MSELoss()
    lr = args.learning_rate
    optimizer = torch.optim.RMSprop(
        pose3d_net.parameters(), lr=args.learning_rate, momentum=0, weight_decay=0
    )

    # 学習データ
    train_dataset = TrainDataset(args, K, phase="train")
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.serial_batches,
        num_workers=args.num_threads,
        pin_memory=args.pin_memory,
    )

    print("train data size: ", len(train_data_loader))

    def set_train():
        pose3d_net.train()

    # training
    start_epoch = 0
    train_start_time = time.time()
    for epoch in range(start_epoch, args.num_epoch):
        set_train()
        for train_idx, train_data in enumerate(train_data_loader):
            # retrieve the data
            # B x 28 x 28 x 28
            gt_hm_tensor = train_data["heatmap"].to(device=cuda)

            # B x 28 x 28 x 28 x 3
            gt_offset_tensor = train_data["offset"].to(device=cuda)

            image_tensor = train_data["img"].to(device=cuda)

            since = time.time()
            hm_tensors, offset_tensors = pose3d_net.forward(image_tensor)
            print("forward time: ", time.time() - since)
            
            # check if hm_tensors in [0, 1]
            hm = hm_tensors.detach().cpu().numpy()

            hm_loss = 4 * hm_criterion(gt_hm_tensor, hm_tensors)
            weight = gt_hm_tensor[:, :, :, :, :, None].expand_as(gt_offset_tensor)
            of_loss = of_criterion(
                gt_offset_tensor * weight,
                offset_tensors * weight,
            )

            loss = hm_loss + of_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_idx % 1 == 0:
                message = f"Epoch: {epoch} | {train_idx}/{len(train_data_loader)} | "
                message += f"Loss: {loss.item()} | Heatmap Loss: {hm_loss.item()} | Offset Loss: {of_loss.item()} | "
                message += f"LR: {lr} | Elapsed {time.time() - train_start_time:.1f}"
                print(message)

            if train_idx == len(train_data_loader) - 1:
                state_dict = pose3d_net.state_dict()
                out_ckpt_path = "%s/epoch_%d.pth" % (
                    args.checkpoints_path,
                    epoch,
                )
                torch.save(state_dict, out_ckpt_path)
                print("Saved", out_ckpt_path)

                # Visualize Result
                gt_hm_tensor_np = gt_hm_tensor.detach().cpu().numpy()
                gt_offset_tensor_np = gt_offset_tensor.detach().cpu().numpy()
                kps = calc_pose3d(gt_hm_tensor_np[0], gt_offset_tensor_np[0])
                plot_pose3d(f"{args.checkpoints_path}/gt_epoch_{epoch}.jpg", kps)

                hm_tensors_np = hm_tensors.detach().cpu().numpy()
                offset_tensors_np = offset_tensors.detach().cpu().numpy()
                kps = calc_pose3d(hm_tensors_np[0], offset_tensors_np[0])
                plot_pose3d(f"{args.checkpoints_path}/pred_epoch_{epoch}.jpg", kps)


if __name__ == "__main__":
    main()
