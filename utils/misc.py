class ScalerMeter(object):

    def __init__(self):
        self.x = None

    def update(self, x):
        if not isinstance(x, (int, float)):
            x = x.item()
        self.x = x

    def reset(self):
        self.x = None

    def get_value(self):
        if self.x:
            return self.x
        return 0


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, x, n=1):
        self.sum += float(x)
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def get_value(self):
        if self.n:
            return self.sum / self.n
        return 0


class MovingAverageMeter(object):

    def __init__(self, decay=0.95):
        self.x = None
        self.decay = decay

    def update(self, x, n=1):
        if n > 0:
            x = float(x) / n
            if self.x is None:
                self.x = x
            else:
                self.x = self.x * self.decay + x * (1 - self.decay)

    def reset(self):
        self.x = None

    def get_value(self):
        if self.x:
            return self.x
        return 0


class PerClassMeter(object):

    def __init__(self, num_classes, meter, **kwargs):
        self.num_classes = num_classes
        self.meters = [meter(**kwargs) for _ in range(num_classes)]

    def update(self, x, y, n=1):
        for i in range(self.num_classes):
            mask = (y == i)
            self.meters[i].update(sum(x[mask]), sum(mask).item())

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def get_value(self, per_class_avg=True):
        values = [meter.get_value() for meter in self.meters]
        if per_class_avg:
            return sum(values) / len(values)
        else:
            return values


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    r"""Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)
