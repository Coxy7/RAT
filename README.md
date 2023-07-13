# Adversarially Robust Source-free Domain Adaptation with Relaxed Adversarial Training

This repository is the official PyTorch implementation of _"Adversarially Robust Source-free Domain Adaptation with Relaxed Adversarial Training"_, accepted by **ICME 2023**.

![pipeline](https://github.com/Coxy7/RAT/assets/22617682/f4cc12f5-d915-4157-9452-bc305fc46d09)

## Setups

### 1. Python environment

- Python 3.9
- PyTorch 1.11
- cudatoolkit 11.3.1
- torchvision 0.12.0
- tensorboard 2.8.0
- scikit-learn 1.0.2
- tqdm
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) 3.2.2
- [robustness](https://github.com/MadryLab/robustness) (only required for using robust pre-trained weights)

### 2. Prepare datasets

The structure of the data directory (`DATA_DIR`) should be like
```
dataset_1
    domain_1
        class_1
            image_1.jpg
            image_2.jpg
            ...
        class_2
        ...
    domain_2
    ...
dataset_2
...
```

Specific names for datasets / domains / classes above are listed in the following table:
|Dataset|`dataset_*`|`domain_*`|`class_*`|
|-|-|-|-|
|Office-31|`office`|`amazon`, `dslr`, `webcam`|`back_pack`, `bookcase`, ...|
|Office-home|`OfficeHome`|`Art`, `Clipart`, `Product`, `RealWorld`|`Alarm_Clock`, `Bottle`, ...|
|PACS|`PACS`|`art_painting`, `cartoon`, `photo`, `sketch`|`dog`, `elephant`, ...|
|VisDA-C|`VisDA`|`train`, `validation`|`aeroplane`, `bicycle`, ...|

Links to datasets: [Office-31](https://www.cc.gatech.edu/~judy/domainadapt), [Office-home](https://www.hemanthdv.org/officeHomeDataset), [PACS](https://dali-dl.github.io/project_iccv2017), [VisDA-C](http://ai.bu.edu/visda-2017)

### 3. Prepare pre-trained models

A directory for pre-trained models (`MODEL_DIR`) is required, which should contain:
- [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) : standard pre-trained ResNet-50, provided by torchvision
- [resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) : standard pre-trained ResNet-101, provided by torchvision

For robust pre-trained weights:
1. Download the ImageNet robust pre-trained weights provided by [Robustness](https://github.com/MadryLab/robustness) package.
    - eps=4/255: [imagenet_linf_4.pt](https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0)
    - eps=8/255: [imagenet_linf_8.pt](https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0)
2. Convert the weights to TorchVision format.
    - Command: `python convert_robustness_to_torchvision.py /path/to/imagenet_linf_8.pt`
    - Put the converted file(s) to `MODEL_DIR`


### 4. Setup directories in `train.sh` and `eval.sh`

Please change line 3-5 of the main scripts `train.sh` and `eval.sh` to proper directories:
- `LOG_ROOT`: root directory for all the logging of experiments
- `DATA_DIR`: the directory for all datasets as stated above
- `MODEL_DIR`: the directory for pre-trained models as stated above

## Code usage

The bash script `train.sh` provides a uniform and simplified interface of the Python scripts for training, which accepts the following arguments:
- eps: adversarial perturbation bound in l_inf (e.g., `4`, `8`, `12`)
- dataset: `office`, `officehome`, `pacs`, or `visda`
- adaptation tasks: can be a list of one or more tasks (see commands below)
- pre-training: standard (`std`) or robust (`rob4` for eps=4/255 or `rob8` for eps=8/255)
- method: the training method (see commands below)
- alpha: the relaxation factor for RAT (ignored for other training methods)
- seed: an integer seed number (note: we use three seeds (0, 1, 2) in the paper)
- other arguments that are passed to the Python scripts (e.g. `--gpu`)

Please refer to the following examples.

### Source model training:
```bash
bash train.sh 8 office 'a2w w2d d2a d2w a2d w2a' std 'SHOT_source' 0 0 --gpu 0
bash train.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' std 'SHOT_source' 0 0 --gpu 0
bash train.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' std 'SHOT_source' 0 0 --gpu 0
bash train.sh 8 visda 't2v' std 'SHOT_source' 0 0 --gpu 0
```

### Teacher model training (SHOT):
```bash
bash train.sh 8 office 'a2w w2d d2a d2w a2d w2a' std 'SHOT' 0 0 --gpu 0
bash train.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' std 'SHOT' 0 0 --gpu 0
bash train.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' std 'SHOT' 0 0 --gpu 0
bash train.sh 8 visda 't2v' std 'SHOT' 0 0 --gpu 0
```

### Adversarial training (AT) with hard pseudo-labels
```bash
bash train.sh 8 office 'a2w w2d d2a d2w a2d w2a' std 'SHOT_PGDAT' 0 0 --gpu 0
bash train.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' std 'SHOT_PGDAT' 0 0 --gpu 0
bash train.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' std 'SHOT_PGDAT' 0 0 --gpu 0
bash train.sh 8 visda 't2v' std 'SHOT_PGDAT' 0 0 --gpu 0
```

### Relaxed Adversarial Training (RAT)

Note: AT with soft labels is equivalent to RAT with alpha=0.

```bash
# Standard pre-training
bash train.sh 8 office 'a2w w2d d2a d2w a2d w2a' std 'SHOT_RAT' 0.1 0 --gpu 0
bash train.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' std 'SHOT_RAT' 0.2 0 --gpu 0
bash train.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' std 'SHOT_RAT' 0.4 0 --gpu 0
bash train.sh 8 visda 't2v' std 'SHOT_RAT' 0.3 0 --gpu 0

# Robust pre-training (ResNet-50 only)
bash train.sh 8 office 'a2w w2d d2a d2w a2d w2a' rob8 'SHOT_RAT' 0.1 0 --gpu 0
bash train.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' rob8 'SHOT_RAT' 0.2 0 --gpu 0
bash train.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' rob8 'SHOT_RAT' 0.4 0 --gpu 0
```

### Evaluation

The script `eval.sh` can be used to evaluate the models with two metrics: accuracy on clean samples and accuracy on adversarial samples (produced by PGD-20 by default). The arguments are exactly the same as those of `train.sh`.

Usage: replace the `train.sh` with `eval.sh` in the above commands for training to evaluate the corresponding models. For example:
```bash
bash eval.sh 8 office 'a2w w2d d2a d2w a2d w2a' std 'SHOT' 0 0 --gpu 0
bash eval.sh 8 officehome 'c2a a2c a2p a2r p2a p2c c2p c2r r2a r2c r2p p2r' std 'SHOT_PGDAT' 0 0 --gpu 0
bash eval.sh 8 pacs 'c2a a2c a2p a2s p2a p2c c2p c2s s2a s2c s2p p2s' std 'SHOT_RAT' 0.4 0 --gpu 0
```
