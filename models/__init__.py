from .resnet import resnet
from .classification import ClassificationModel, SHOTModel


__all__ = ['get_model']


def get_model(arch, num_classes, preprocess_fn, variant='std', dim=None, small_image=False,is_source =False):
    if arch.startswith('resnet'):
        backbone, head = resnet(
            arch, num_classes, variant=variant, dim=dim, small_image=small_image)
    else:
        raise NotImplementedError()
    if variant == 'std':
        model = ClassificationModel(
            num_classes=num_classes,
            backbone=backbone,
            classifier=head,
            preprocess_fn=preprocess_fn,
        )
    elif variant == 'shot':
        model = SHOTModel(
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            preprocess_fn=preprocess_fn,
        )
    return model
