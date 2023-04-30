import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.trainer import ClassificationTrainer


class StandardTrainer(ClassificationTrainer):

    def __init__(self, manager, models, dataloaders, criterions,
                 optimizers, schedulers, schedule_steps, num_epochs,
                 iters_per_epoch, steps_per_iter, accumulation_steps,
                 device, attack=None, eval_attack=None,
                 pseudo_labels=None, acc_per_class=False, grad_clip=None):
        self.opt = manager.get_opt()
        super().__init__(
            manager=manager,
            models=models,
            dataloaders=dataloaders,
            criterions=criterions,
            optimizers=optimizers,
            schedulers=schedulers,
            schedule_steps=schedule_steps,
            num_epochs=num_epochs,
            iters_per_epoch=iters_per_epoch,
            phases_per_iter=1,
            steps_per_iter=steps_per_iter,
            accumulation_steps=accumulation_steps,
            log_period=self.opt.log_period,
            ckpt_period=self.opt.ckpt_period,
            test_period=self.opt.test_period,
            device=device,
            test_no_grad=False,
        )
        self.add_meter('learning_rate', 'lr', meter_type='scaler')
        self.add_meter('loss', 'L', fstr_format='6.3f')
        self.log_clean_acc = self.opt.log_clean_acc
        mtype = 'per_class_avg' if acc_per_class else 'avg'
        n_class = models[0].num_classes
        self.add_meter('acc_train_clean', 'TrCl', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        if attack or eval_attack:
            self.add_meter('acc_train_adv', 'TrAd', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        self.add_meter('acc_test_clean', 'TeCl', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        if attack or eval_attack:
            self.add_meter('acc_test_adv', 'TeAd', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        self.attack = attack
        self.eval_attack = eval_attack
        self.pseudo_labels = pseudo_labels
        self.grad_clip = grad_clip

    def get_data_batch(self, phase):
        if phase == 0:
            (images, labels), idx = self._next_data_batch(0)
            batch = (images, labels, idx)
        else:
            (train_images, train_labels), _ = self._next_data_batch(1)
            (test_images, test_labels), _ = self._next_data_batch(2)
            batch = (train_images, train_labels, test_images, test_labels)
        return [t.to(self.device) for t in batch]

    def get_active_optimizers(self, phase):
        if phase == 0:
            return self.optimizers
        else:
            return []

    def get_checkpoint(self, epoch_id):
        model, = self.models
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch_id,
        }
        return checkpoint

    def toggle_grad(self, phase):
        pass
        # model, = self.models
        # for name, param in model.parameters():
        #     param.requires_grad = (phase == 0)

    def update_meters(self, training):
        scheduler, = self.schedulers
        self.meters['learning_rate'].update(scheduler.get_last_lr()[0])

    def do_step(self, phase, iter_id, epoch_id, data_batch):
        if phase == 0:
            self.do_step_train(data_batch=data_batch)
        else:
            with torch.no_grad():
                self.do_step_test(data_batch=data_batch)

    def do_step_train(self, data_batch):
        model, = self.models 
        criterion_cls = self.criterions[0]
        images, labels, idx = data_batch
        true_labels = labels

        if self.pseudo_labels is not None:
            labels = self.pseudo_labels[idx]

        if self.log_clean_acc:
            with torch.no_grad():
                logits_clean = model(images)
                pred_clean = logits_clean.argmax(1)
                # loss = criterion_cls(logits_clean, labels)

        attack = self.attack
        if attack:
            images_adv = attack(images, labels)
            logits_adv = model(images_adv)
            pred_adv = logits_adv.argmax(1)
            if self.opt.at_mode == 'pgd_at':
                loss = criterion_cls(logits_adv, labels)
            elif self.opt.at_mode == 'trades':
                logits_clean = model(images)
                criterion_trades = self.criterions[1]
                loss = criterion_cls(logits_clean, labels)
                loss += self.opt.beta_trades * criterion_trades(logits_clean, logits_adv)
            else:
                loss = 0
        else:
            logits_clean = model(images)
            pred_clean = logits_clean.argmax(1)
            loss = criterion_cls(logits_clean, labels)

        loss /= self.accumulation_steps[0]
        loss.backward()

        self.meters['loss'].update(loss)
        if self.log_clean_acc or attack is None:
            self._update_acc_meter('acc_train_clean', pred_clean, true_labels)
        if attack:
            self._update_acc_meter('acc_train_adv', pred_adv, true_labels)

    def do_step_test(self, data_batch):
        model = self.models[0]
        train_images, train_labels, test_images, test_labels = data_batch

        pred_train_clean = model(train_images).argmax(1)
        self._update_acc_meter('acc_train_clean', pred_train_clean, train_labels)
        pred_test_clean = model(test_images).argmax(1)
        self._update_acc_meter('acc_test_clean', pred_test_clean, test_labels)

        if self.eval_attack:
            with torch.enable_grad():
                images_adv = self.eval_attack(train_images, train_labels)
            pred_adv = model(images_adv).argmax(1)
            self._update_acc_meter('acc_train_adv', pred_adv, train_labels)
            with torch.enable_grad():
                images_adv = self.eval_attack(test_images, test_labels)
            pred_adv = model(images_adv).argmax(1)
            self._update_acc_meter('acc_test_adv', pred_adv, test_labels)
