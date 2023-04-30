import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.standard_trainer import StandardTrainer


class DistillTrainer(StandardTrainer):

    def __init__(self, manager, models, dataloaders, criterions,
                 optimizers, schedulers, schedule_steps, num_epochs,
                 iters_per_epoch, steps_per_iter, accumulation_steps,
                 device, augment=None, attack=None, eval_attack=None,
                 teacher=None, acc_per_class=False, grad_clip=None):
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
            steps_per_iter=steps_per_iter,
            accumulation_steps=accumulation_steps,
            device=device,
            attack=attack,
            eval_attack=eval_attack,
            acc_per_class=acc_per_class,
            grad_clip=grad_clip,
        )
        assert teacher is not None
        self.augment = augment
        self.teacher = teacher

    def get_data_batch(self, phase):
        if phase == 0:
            batch = self._next_data_batch(0)
        else:
            train_images, train_labels = self._next_data_batch(1)
            test_images, test_labels = self._next_data_batch(2)
            batch = (train_images, train_labels, test_images, test_labels)
        return [t.to(self.device) for t in batch]

    def do_step_train(self, data_batch):
        model, = self.models 
        criterion_cls = self.criterions[0]
        images, true_labels = data_batch
        if self.augment is not None:
            images = self.augment(images)
        with torch.no_grad():
            soft_labels = self.teacher(images)
        pseudo_labels = soft_labels.argmax(1)

        if self.log_clean_acc:
            with torch.no_grad():
                logits_clean = model(images)
                pred_clean = logits_clean.argmax(1)
                # loss = criterion_cls(logits_clean, labels)

        attack = self.attack
        if attack:
            images_adv = attack(images, pseudo_labels)
            logits_adv = model(images_adv)
            pred_adv = logits_adv.argmax(1)
            if self.opt.at_mode == 'pgd_at':
                loss = criterion_cls(logits_adv, soft_labels)
            elif self.opt.at_mode == 'trades':
                logits_clean = model(images)
                criterion_trades = self.criterions[1]
                loss = criterion_cls(logits_clean, soft_labels)
                loss += self.opt.beta_trades * criterion_trades(logits_clean, logits_adv)
            elif self.opt.at_mode == 'rat':
                with torch.no_grad():
                    soft_labels_adv = self.teacher(images_adv)
                alpha = self.opt.rat_alpha
                relaxed_soft_labels = (1 - alpha) * soft_labels + alpha * soft_labels_adv
                loss = criterion_cls(logits_adv, relaxed_soft_labels)
        else:
            logits_clean = model(images)
            pred_clean = logits_clean.argmax(1)
            loss = criterion_cls(logits_clean, soft_labels)

        loss /= self.accumulation_steps[0]
        loss.backward()

        self.meters['loss'].update(loss.item())
        if self.log_clean_acc or attack is None:
            self._update_acc_meter('acc_train_clean', pred_clean, true_labels)
        if attack:
            self._update_acc_meter('acc_train_adv', pred_adv, true_labels)

    def do_step_test(self, data_batch):
        model = self.models[0]
        train_images, train_labels, test_images, test_labels = data_batch
        if self.opt.val_split:
            with torch.no_grad():
                train_labels = self.teacher(train_images).argmax(1)
                test_labels = self.teacher(test_images).argmax(1)

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
