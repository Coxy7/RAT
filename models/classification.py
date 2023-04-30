import torch.nn as nn
import torch

class ClassificationModel(nn.Module):

    def __init__(self, num_classes, backbone, classifier, preprocess_fn=None):
        super().__init__()
        self.num_classes = num_classes
        self.preprocess_fn = preprocess_fn
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, images, get_features=False, **kwargs):
        x = self.preprocess_fn(images)
        f = self.backbone(x, **kwargs)
        y = self.classifier(f)
        if get_features:
            return y, f
        else:
            return y

    def get_features(self, images, **kwargs):
        x = self.preprocess_fn(images)
        f = self.backbone(x, **kwargs)
        return f


class SHOTModel(nn.Module):

    def __init__(self, num_classes, backbone, head, preprocess_fn=None):
        super().__init__()
        self.num_classes = num_classes
        self.preprocess_fn = preprocess_fn
        self.backbone = backbone
        self.bottleneck = head[0]
        self.classifier = head[1]

    def forward(self, images, get_features=False, **kwargs):
        x = self.preprocess_fn(images)
        b = self.backbone(x, **kwargs)
        f = self.bottleneck(b)
        y = self.classifier(f)
        if get_features:
            return y, f
        else:
            return y

    def get_features(self, images, **kwargs):
        x = self.preprocess_fn(images)
        b = self.backbone(x, **kwargs)
        f = self.bottleneck(b)
        return f
