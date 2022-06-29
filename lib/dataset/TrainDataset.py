from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch
from PIL.ImageFilter import GaussianBlur
import logging
from pathlib import Path
import cv2


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, args, phase="train"):
        self.opt = args

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, "RENDER")
        self.PARAMS = os.path.join(self.root, "PARAMS")
        self.is_train = phase == "train"
        self.input_size = self.opt.input_size
        self.subjects = self.get_subjects()
        self.frame_num = self.opt.frame_num
        self.frame_list = list(range(self.frame_num - 2))

        # PIL to tensor
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(self.opt.load_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # augmentation
        self.aug_trans = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.05,
                    hue=0.05,
                )
            ]
        )

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        return all_subjects

    def __len__(self):
        return len(self.subjects) * (self.frame_num - 2)

    def load_heat_maps(self, param_path):
        n_grid = 28
        sigma = 2

        hm = np.zeros((24, n_grid, n_grid, n_grid))
        offset = np.zeros((24, n_grid, n_grid, n_grid, 3))

        i_kp = 0
        with open(param_path) as f:
            line = f.readline()
            while line:
                tokens = line.split(" ")
                if len(tokens) >= 3:
                    kp = np.array([float(v) for v in tokens[:3]])
                    x, y, z = (1 + kp) / 2 * n_grid

                    for i in range(n_grid):
                        for j in range(n_grid):
                            for k in range(n_grid):
                                hm[i_kp, i, j, k] = np.exp(
                                    -((x - i) ** 2 + (y - j) ** 2 + (k - z) ** 2)
                                    / (2 * sigma**2)
                                ) / (2 * np.pi * sigma**2)
                                offset[i_kp, i, j, k, 0] = x - (i + 0.5)
                                offset[i_kp, i, j, k, 1] = y - (j + 0.5)
                                offset[i_kp, i, j, k, 2] = z - (k + 0.5)

                    i_kp += 1
                line = f.readline()

        return hm, offset

    def get_render(self, subject, start_frame):
        # The ids are an even distribution of num_views around view_id
        renders = []
        for i_frame in range(start_frame, start_frame + 3):

            render_path = os.path.join(self.RENDER, subject, "%05d.jpg" % i_frame)
            render = Image.open(render_path).convert("RGB")

            # if self.is_train:
            # TODo
            # Pad images
            # random flip
            # random scale
            # random translate in the pixel space

            # 色味を変える
            # render = self.aug_trans(render)

            # # random blur
            # if self.opt.aug_blur > 0.00001:
            #     blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
            #     render = render.filter(blur)

            renders.append(self.to_tensor(render))

        # list of (3x448x448) -> 9 x 448 x 448
        renders = torch.cat(renders, dim=0)

        # hm: 24 x 28 x 28 x 28, voxel -> 存在確率
        # of: 24 x 28 x 28 x 28 x 3, voxel -> xyz方向の差分（voxel中心からどのくらいずれるか）
        param_path = os.path.join(self.PARAMS, subject, "%05d.txt" % (1 + start_frame))
        hm, of = self.load_heat_maps(param_path)

        # hm: 24 x 28 x 28 x 28, voxel -> 存在確率
        # of: 24 x 28 x 28 x 28 x 3, voxel -> xyz方向の差分（voxel中心からどのくらいずれるか）
        hm = torch.Tensor(hm).float()
        of = torch.Tensor(of).float()

        return {
            "img": renders,
            "heatmap": hm,
            "offset": of,
        }

    def get_item(self, index):
        subject_id = index // (self.frame_num - 2)
        i_frame = index % (self.frame_num - 2)
        subject = self.subjects[subject_id]
        res = {
            "subject_id": subject_id,
            "i_frame": i_frame,
        }
        render_data = self.get_render(subject, i_frame)
        res.update(render_data)
        return res

    def __getitem__(self, index):
        return self.get_item(index)
