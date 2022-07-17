import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, args, K, phase="train"):
        self.opt = args
        self.K = K

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, "RENDER")
        self.PARAMS = os.path.join(self.root, "PARAMS")
        self.is_train = phase == "train"
        self.input_size = self.opt.input_size
        self.subjects, self.subject_frames = self.get_subjects()
        self.n_data = int(np.sum([v for _, v in self.subject_frames.items()]))
        assert self.n_data > 0, "No data found"

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
        subject_frames = {
            (i, s): len(list((Path(self.RENDER) / s).glob("*.jpg"))) for i, s in enumerate(all_subjects)
        }
        subject_frames = {k: n for k, n in subject_frames.items() if n > 3}
        return all_subjects, subject_frames

    def __len__(self):
        return self.n_data

    def load_heat_maps(self, param_path):
        n_grid = 28
        sigma = 2

        hm = np.zeros((self.K, n_grid, n_grid, n_grid))
        offset = np.zeros((self.K, n_grid, n_grid, n_grid, 3))

        i_kp = 0

        bbox = np.array([-1.3, 0, 4.3]), np.array([1.3, 2, 7.5])

        R = 4
        with open(param_path) as f:
            line = f.readline()
            while line:
                tokens = line.split(",")
                if len(tokens) >= 3:
                    kp = np.array([float(v) for v in tokens[:3]])

                    # workaround
                    x, y, z = (kp - bbox[0]) / (bbox[1] - bbox[0]) * n_grid
                    # print(x, y, z)

                    # CHWの順にしたい
                    for i in range(n_grid):
                        for j in range(n_grid):
                            for k in range(n_grid):
                                dz = z - (i + 0.5)
                                dy = y - (j + 0.5)
                                dx = x - (k + 0.5)
                                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                                hm[i_kp, i, j, k] = 1 if dist <= R else 0
                                offset[i_kp, i, j, k, 0] = dz
                                offset[i_kp, i, j, k, 1] = dy
                                offset[i_kp, i, j, k, 2] = dz

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

            # TODO: BG

            renders.append(self.to_tensor(render))

        # list of (3x448x448) -> 9 x 448 x 448
        renders = torch.cat(renders, dim=0)

        # hm: K x 28 x 28 x 28, voxel -> 存在確率
        # of: K x 28 x 28 x 28 x 3, voxel -> xyz方向の差分（voxel中心からどのくらいずれるか）
        param_path = os.path.join(self.PARAMS, subject, "%05d.txt" % (1 + start_frame))
        hm, of = self.load_heat_maps(param_path)

        # hm: K x 28 x 28 x 28, voxel -> 存在確率
        # of: K x 28 x 28 x 28 x 3, voxel -> xyz方向の差分（voxel中心からどのくらいずれるか）
        hm = torch.Tensor(hm).float()
        of = torch.Tensor(of).float()

        return {
            "img": renders,
            "heatmap": hm,
            "offset": of,
        }

    def get_item(self, index):
        sum = 0
        for (i, _), n in self.subject_frames.items():
            sum += n
            if sum > index:
                subject_id = i
                i_frame = index - (sum - n)
                break

        res = {
            "subject_id": subject_id,
            "i_frame": i_frame,
        }
        subject = self.subjects[subject_id]
        
        render_data = self.get_render(subject, i_frame)
        res.update(render_data)
        return res

    def __getitem__(self, index):
        return self.get_item(index)
