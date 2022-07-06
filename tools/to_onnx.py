import torch
import torchvision
import argparse
from pathlib import Path
from lib.model.Pose3DNet import K, Pose3DNet

DEFAULT_CKPT_PATH = r"D:\workspace\InstantPose3D\train_mini\ckpt\epoch_135.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()
dummy_input = torch.randn(1, 9, 448, 448, device="cuda")
pose3d_net = Pose3DNet(args, is_train=False).cuda()


input_names = ["input"]
output_names = ["out_heatmap", "out_offset"]

torch.onnx.export(
    pose3d_net,
    dummy_input,
    "pose3d.onnx",
    export_params=True,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    opset_version=11
)

print("====")

import onnx
# Load the ONNX model
model = onnx.load("pose3d.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

print("OK")
