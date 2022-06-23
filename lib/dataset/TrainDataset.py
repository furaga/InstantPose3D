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
from lib.model.Pose3DNet import Pose3DNet
from lib.dataset.TrainDataset import TrainDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, default=None)
    parser.add_argument("--checkpoints_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--name", type=str, default="nmlF")

    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--train_yaw_step", type=int, default=1)

    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--output_size", type=int, default=228)
    # TODO: augment系

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # set cuda
    cuda = torch.device("cuda:%d" % args.gpu_id)

    # Setup network
    # https://www.youtube.com/watch?v=kKNcyTqybjA&ab_channel=%E9%9D%92%E6%9F%B3%E5%B9%B8%E5%BD%A6
    pose3d_net = Pose3DNet(args.input_size, args.output_size).to(device=cuda)

    if args.pretrained_path is not None:
        print("=============")
        print("loading pretrained model ...", args.pretrained_path)
        print("=============")
        state_dict = torch.load(args.pretrained_path, map_location=cuda)
        pose3d_net.load_state_dict(state_dict)

    # Create output checkpionts directory
    args.checkpoints_path.mkdir(exist_ok=True, parents=True)

    # 学習パラメタ: TODO
    criterion = torch.nn.L2Loss()
    lr = args.learning_rate
    optimizer = torch.optim.RMSprop(
        pose3d_net.parameters(), lr=args.learning_rate, momentum=0, weight_decay=0
    )

    # 学習データ
    train_dataset = TrainDataset(args, phase="train")
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
    for epoch in range(start_epoch, args.num_epoch):
        set_train()
        for train_idx, train_data in enumerate(train_data_loader):
            # retrieve the data
            gt_hm0_tensor = train_data["heat_map_L0"].to(device=cuda)
            gt_hm1_tensor = train_data["heat_map_L1"].to(device=cuda)

            image_tensor = train_data["img"].to(device=cuda)
            hm0_tensor, hm1_tensor = pose3d_net.forward(image_tensor)

            loss0 = criterion(hm0_tensor, gt_hm0_tensor)
            loss1 = criterion(hm1_tensor, gt_hm1_tensor)
            err = loss0 + loss1

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if train_idx % args.freq_plot == 0:
                print(
                    "Type: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f}".format(
                        args.type,
                        epoch,
                        train_idx,
                        len(train_data_loader),
                        err.item(),
                        lr,
                    )
                )

            if train_idx == len(train_data_loader) - 1:
                state_dict = pose3d_net.state_dict()
                out_ckpt_path = "%s/epoch_%d.pth" % (
                    args.checkpoints_path,
                    epoch,
                )
                torch.save(state_dict, out_ckpt_path)
                print("Saved", out_ckpt_path)

                save_combined = np.hstack(save_imgs)
                cv2.imwrite(f"{epoch}.png", save_combined)


if __name__ == "__main__":
    main()
