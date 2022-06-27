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
from lib.model.Pose3DNet import Pose3DNet, hg_layer_nums
from lib.dataset.TrainDataset import TrainDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, default=None)
    parser.add_argument("--checkpoints_path", type=Path, required=True)

    # 学習パラメタ
    parser.add_argument("--batch_size", type=int, default=1)
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
            # B x 28 x 28 x 28
            gt_hm_tensor = train_data["heatmap"].to(device=cuda)
            # B x 28 x 28 x 28 x 3
            gt_offset_tensor = train_data["offset"].to(device=cuda)

            image_tensor = train_data["img"].to(device=cuda)
            hm_tensors, offset_tensors = pose3d_net.forward(image_tensor)
            # list of Bxchx28x28
            # list of Bxchx28x28x3

            loss = 0
            last_ch = hg_layer_nums[-1]
            for i, ch in enumerate(hg_layer_nums):
                gt_inter_hm = torch.reshape(
                    gt_hm_tensor, (-1, ch, last_ch // ch, -1, -1)
                )
                gt_inter_hm = torch.mean(gt_inter_hm, dim=2)
                loss += criterion(gt_inter_hm, hm_tensors[i])

                gt_inter_of = torch.reshape(
                    gt_offset_tensor, (-1, ch, last_ch // ch, -1, -1, -1)
                )
                gt_inter_of = torch.mean(gt_inter_of, dim=2)
                loss += criterion(gt_inter_of, offset_tensors[i])

            err = loss

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


if __name__ == "__main__":
    main()
