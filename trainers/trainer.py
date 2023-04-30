import os
from tqdm import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler

# from utils.experiman import manager
from utils.misc import AverageMeter, MovingAverageMeter, ScalerMeter, PerClassMeter


class BaseTrainer():
    """
    Phases:
        (0, self.phases_per_iter): train
        -1: test
    """

    def __init__(self, manager, models, dataloaders, criterions,
                 optimizers, schedulers, schedule_steps, num_epochs,
                 iters_per_epoch, phases_per_iter, steps_per_iter,
                 accumulation_steps, log_period, ckpt_period, device,
                 test_period=1, test_no_grad=True):
        self.manager = manager
        self.is_master = manager.is_master()
        if self.is_master:
            self.logger = manager.get_logger()
            self.msg = ''
            self.last_log_iter_id = -1 
            self.tqdms = [None for _ in iters_per_epoch]
        self.models = models
        self.dataloaders = dataloaders
        self.data_iters = [iter(loader) for loader in dataloaders]
        self.data_counters = [0 for _ in dataloaders]
        for loader in self.dataloaders:
            if isinstance(loader._index_sampler, DistributedSampler):
                loader._index_sampler.set_epoch(0)
        self.criterions = criterions
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.schedule_steps = schedule_steps
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.steps_per_iter = steps_per_iter
        self.phases_per_iter = phases_per_iter
        self.accumulation_steps = accumulation_steps
        self.log_period = log_period
        self.ckpt_period = ckpt_period
        self.test_period = test_period
        self.device = device
        self.test_no_grad = test_no_grad
        self.meters = {}
        self.meters_info = {}
        self.iter_count = 0

    def _setup_tqdms(self, training):
        idx = 0 if training else 1
        t = tqdm(total=self.iters_per_epoch[idx], leave=True, dynamic_ncols=True)
        t.clear()
        self.tqdms[idx] = t

    def _next_data_batch(self, idx):
        try:
            batch = next(self.data_iters[idx])
        except StopIteration:
            self.data_counters[idx] += 1
            loader = self.dataloaders[idx]
            if isinstance(loader._index_sampler, DistributedSampler):
                loader._index_sampler.set_epoch(self.data_counters[idx])
            self.data_iters[idx] = iter(loader)
            batch = next(self.data_iters[idx])
        return batch

    def get_data_batch(self, phase):
        """Return a batch of data for the phase."""
        raise NotImplementedError

    def get_active_optimizers(self, phase):
        """Return the optimizers active for the phase."""
        raise NotImplementedError

    def get_checkpoint(self, epoch_id):
        """Return a checkpoint object to be saved."""
        raise NotImplementedError

    def toggle_grad(self, phase):
        """Turn on/off the gradient computation for models."""
        
    def update_meters(self, training):
        """Update meters before logging."""

    def do_step(self, phase, iter_id, epoch_id, data_batch):
        """
        Typical procedure:
        1. Forward for losses;
        2. Backward for gradients (if needed);
        3. Update (avg/sum) meters.
        """
        raise NotImplementedError

    def update_lr(self, step):
        for scheduler, schedule_step in zip(self.schedulers, self.schedule_steps):
            if schedule_step == step:
                scheduler.step()

    def do_iter(self, iter_id, epoch_id, training):
        if training:
            # data_batch = self.get_data_batch(0)
            for phase in range(self.phases_per_iter):
                self.toggle_grad(phase)
                for _ in range(self.steps_per_iter[phase]):
                    optimizers = self.get_active_optimizers(phase)
                    # for optimizer in optimizers:
                    #     optimizer.zero_grad()
                    for _ in range(self.accumulation_steps[phase]):
                        data_batch = self.get_data_batch(phase)
                        self.do_step(phase, iter_id, epoch_id, data_batch)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
            self.iter_count += 1
        else:
            phase = -1
            self.toggle_grad(phase)
            data_batch = self.get_data_batch(phase)
            self.do_step(phase, iter_id, epoch_id, data_batch)

    def log_iter(self, iter_id, epoch_id, training):
        self.update_meters(training)
        # Progress bar
        disp_items = [""]
        for name, info in self.meters_info.items():
            fmt = info['format']
            if fmt is not None:
                value = self.meters[name].get_value()
                disp_items.append(f"{info['abbr']} {value:{fmt}}")
        self.msg = '|'.join(disp_items)
        # self.msg = ' | '.join(disp_items)
        idx = 0 if training else 1
        self.tqdms[idx].set_postfix_str(self.msg)
        self.tqdms[idx].update(iter_id - self.last_log_iter_id)
        self.last_log_iter_id = iter_id

        # Third-party tools
        if training:
            for name, meter in self.meters.items():
                self.manager.log_metric(name, meter.get_value(),
                                        self.iter_count, epoch_id, split='train')
    
    def do_epoch(self, epoch_id, training):
        idx = 0 if training else 1
        iters_per_epoch = self.iters_per_epoch[idx]
        for model in self.models:
            model.train(training)
        self.setup_logging(training)
        for iter_id in range(iters_per_epoch):
            self.do_iter(iter_id, epoch_id, training)
            if self.is_master and ((iter_id + 1) % self.log_period == 0 or
                                   iter_id == iters_per_epoch - 1):
                self.log_iter(iter_id, epoch_id, training)
            if training:
                self.update_lr('iter')

    def save_checkpoint(self, epoch_id, checkpoint_name=None):
        checkpoint = self.get_checkpoint(epoch_id)
        if checkpoint_name is None:
            name = f'ckpt-{epoch_id}.pt'
        else:
            name = f'{checkpoint_name}.pt'
        model_path = os.path.join(self.manager.get_checkpoint_dir(), name)
        torch.save(checkpoint, model_path)
        self.logger.info(f'Model saved to: {model_path}')

    def log_epoch(self, epoch_id, training):
        idx = 0 if training else 1
        self.last_log_iter_id = -1
        self.tqdms[idx].close()
        self.update_meters(training)
        if training:
            self.logger.info(f"train: {self.msg}")
        else:
            self.logger.info(f"test: {self.msg}")
            if (epoch_id + 1) % self.ckpt_period == 0:
                self.save_checkpoint(epoch_id)
            for name, meter in self.meters.items():
                self.manager.log_metric(name, meter.get_value(),
                                   self.iter_count, epoch_id, split='val')

    def setup_logging(self, training):
        # meter_dict = {
        #     'avg': MovingAverageMeter if training else AverageMeter,
        #     'scaler': ScalerMeter,
        # }
        for name, info in self.meters_info.items():
            meter_type = info['type']
            if meter_type == 'avg':
                # if training:
                if False:
                    meter = MovingAverageMeter()
                else:
                    meter = AverageMeter()
            elif meter_type == 'per_class_avg':
                # if training:
                if False:
                    meter = PerClassMeter(info['num_classes'], MovingAverageMeter)
                else:
                    meter = PerClassMeter(info['num_classes'], AverageMeter)
            elif meter_type == 'scaler':
                meter = ScalerMeter()
            else:
                raise NotImplementedError()
            self.meters[name] = meter


    def add_meter(self, name, abbr=None, meter_type='avg', fstr_format=None, num_classes=None):
        assert meter_type in ('avg', 'scaler', 'per_class_avg')
        self.meters_info[name] = {
            'abbr': abbr if abbr is not None else name,
            'type': meter_type,
            'format': fstr_format,
            'num_classes': num_classes,
        }

    def train(self):
        for epoch_id in range(self.num_epochs):
            if self.is_master:
                lrs = [scheduler.get_last_lr()[0] for scheduler in self.schedulers]
                lrs = "|".join([f"{lr:.5f}" for lr in lrs])
                self.logger.info(f'Epoch: {epoch_id}/{self.num_epochs} lr: {lrs}')
            if (epoch_id + 1) % self.test_period == 0 or \
                    epoch_id == self.num_epochs - 1:
                stages = [True, False]
            else:
                stages = [True]
            for training in stages:
                if self.is_master:
                    self._setup_tqdms(training)
                if not training and self.test_no_grad:
                    with torch.no_grad():
                        self.do_epoch(epoch_id, training)
                else:
                    self.do_epoch(epoch_id, training)
                if self.is_master:
                    self.log_epoch(epoch_id, training)
            self.update_lr('epoch')
        if self.is_master:
            self.save_checkpoint(self.num_epochs - 1, 'ckpt')


class ClassificationTrainer(BaseTrainer):

    def __init__(self, manager, models, dataloaders, criterions, optimizers, schedulers, schedule_steps, num_epochs, iters_per_epoch, phases_per_iter, steps_per_iter, accumulation_steps, log_period, ckpt_period, device, test_period=1, test_no_grad=True):
        super().__init__(manager, models, dataloaders, criterions, optimizers, schedulers, schedule_steps, num_epochs, iters_per_epoch, phases_per_iter, steps_per_iter, accumulation_steps, log_period, ckpt_period, device, test_period=test_period, test_no_grad=test_no_grad)

    def _update_acc_meter(self, meter_name, predictions, labels):
        if self.meters_info[meter_name]['type'] == 'per_class_avg':
            self.meters[meter_name].update(
                predictions.eq(labels), labels)
        else:
            self.meters[meter_name].update(
                predictions.eq(labels).sum(), labels.size(0))
