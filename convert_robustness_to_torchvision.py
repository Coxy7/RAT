import os
import argparse
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_in', type=str)
    opt = parser.parse_args()
    path, name = os.path.split(opt.file_in)
    name = 'resnet50-' + name + 'h'
    file_out = os.path.join(path, name)
    print(f"==> Converting: {opt.file_in}")

    model, _ = make_and_restore_model(
        arch='resnet50', dataset=ImageNet(''), resume_path=opt.file_in)
    d = model.model.state_dict()
    torch.save(d, file_out)
    print(f"==> Converted file saved to: {file_out}")

if __name__ == "__main__":
    main()
