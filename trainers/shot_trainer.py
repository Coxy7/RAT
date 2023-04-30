import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


from trainers.trainer import ClassificationTrainer


class SHOTTrainer(ClassificationTrainer):

    def __init__(self, manager, models, dataloaders, criterions,
                 optimizers, schedulers, schedule_steps, num_epochs,
                 iters_per_epoch, steps_per_iter, accumulation_steps,
                 device, attack=None, eval_attack=None,
                 pseudo_labels=None, cls_loss_start_epoch=0,
                 acc_per_class=False, grad_clip=None):
        """
        Args:
            models: (model,)
            dataloaders: (train_aug, train_raw, test)
            criterions: ([CE, IMLoss])
            optimizers: (the optimizer,)
        """ 
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
        self.log_clean_acc = self.opt.log_clean_acc
        mtype = 'per_class_avg' if acc_per_class else 'avg'
        n_class = models[0].num_classes
        self.add_meter('loss', 'L', meter_type='avg', fstr_format='6.3f')
        self.add_meter('acc_train_clean', 'TrCl', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        if eval_attack:
            self.add_meter('acc_train_adv', 'TrAd', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        self.add_meter('acc_test_clean', 'TeCl', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        if eval_attack:
            self.add_meter('acc_test_adv', 'TeAd', meter_type=mtype, fstr_format='5.3f', num_classes=n_class)
        self.attack = attack
        self.eval_attack = eval_attack
        self.pseudo_labels = pseudo_labels
        self.fixed_pseudo_labels = (pseudo_labels is not None)
        self.cls_loss_start_epoch = cls_loss_start_epoch
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
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_id,
        }
        return checkpoint

    def toggle_grad(self, phase):
        pass
        # model = self.models[0]
        # for param in model.parameters():
        #     param.requires_grad = (phase == 0)

    def update_meters(self, training):
        scheduler = self.schedulers[0]
        self.meters['learning_rate'].update(scheduler.get_last_lr()[0])

    def train(self):
        """Overridden"""
        for epoch_id in range(self.num_epochs):
            if self.is_master:
                lrs = [scheduler.get_last_lr()[0] for scheduler in self.schedulers]
                lrs = "|".join([f"{lr:.5f}" for lr in lrs])
                self.logger.info(f'Epoch: {epoch_id}/{self.num_epochs} lr: {lrs}')
            if not self.fixed_pseudo_labels:
                self.update_pseudo_labels()
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

    def do_step(self, phase, iter_id, epoch_id, data_batch):
        if phase == 0:
            self.do_step_train(epoch_id, data_batch=data_batch)
        else:
            with torch.no_grad():
                self.do_step_test(data_batch=data_batch)

    def do_step_train(self, epoch_id, data_batch):
        model, = self.models
        images, labels, idx = data_batch
        pseudo_labels = self.pseudo_labels[idx]
        criterion_cls = self.criterions[0]
        criterion_im = self.criterions[1]

        if self.log_clean_acc:
            with torch.no_grad():
                logits_clean = model(images)
                pred_clean = logits_clean.argmax(1)
                # loss = criterion_cls(logits_clean, labels)

        if self.attack:
            images_adv = self.attack(images, pseudo_labels)
            logits_adv, embeddings = model(images_adv, get_features=True)
            pred_adv = logits_adv.argmax(1)
            if self.opt.at_mode == 'pgd_at':
                loss = criterion_im(logits_adv) * self.opt.lambda_im
                loss += criterion_cls(logits_adv, pseudo_labels) * self.opt.lambda_cls
            elif self.opt.at_mode == 'trades':
                logits_clean = model(images)
                criterion_trades = self.criterions[2]
                loss = criterion_im(logits_clean) * self.opt.lambda_im
                loss += criterion_cls(logits_clean, pseudo_labels) * self.opt.lambda_cls
                loss += self.opt.beta_trades * criterion_trades(logits_clean, logits_adv)
        else:
            logits_clean, embeddings = model(images, get_features=True)
            pred_clean = logits_clean.argmax(1)
            loss = criterion_im(logits_clean) * self.opt.lambda_im
            if epoch_id >= self.cls_loss_start_epoch:
                loss += criterion_cls(logits_clean, pseudo_labels) * self.opt.lambda_cls
        
        loss /= self.accumulation_steps[0]
        loss.backward()

        self.meters['loss'].update(loss.item())

        if self.log_clean_acc or self.attack is None:
            self._update_acc_meter('acc_train_clean', pred_clean, labels)
        if self.attack:
            self._update_acc_meter('acc_train_adv', pred_adv, labels)

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

    def update_pseudo_labels(self):
        model, = self.models
        training = model.training
        model.eval()

        logits, features, labels, idx = self._evaluate(self.dataloaders[1])
        prediction_model = torch.argmax(logits, dim=1)
        prediction_cluster = self._cluster(logits, features)

        acc_model = prediction_model.eq(labels).float().mean().item()
        acc_cluster = prediction_cluster.eq(labels).float().mean().item()
        self.pseudo_labels = torch.empty_like(prediction_cluster)
        self.pseudo_labels[idx] = prediction_cluster
        self.logger.info(f"Model / clustering accuracy: {acc_model:5.3f} / {acc_cluster:5.3f}")
        model.train(training)

    def _evaluate(self, dataloader):
        model, = self.models
        features_list = []
        logits_list = []
        labels_list = []
        idx_list = []
        for (images, labels), idx in dataloader:
        # for (images, labels), idx in tqdm(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                logits, features = model(images, get_features=True)
            logits_list.append(logits)
            features_list.append(features)
            labels_list.append(labels)
            idx_list.append(idx)
        return [torch.cat(l, 0) for l in (logits_list, features_list, labels_list, idx_list)]

    def _cluster(self, all_output, all_fea):
        all_output = nn.Softmax(dim=1)(all_output)
        predict = torch.argmax(all_output, 1).cpu().numpy()

        # cosine distance
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).to(self.device)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > self.opt.threshold)[0]

        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(1):
            cls_count = np.eye(K)[pred_label].sum(axis=0)
            labelset = np.where(cls_count > self.opt.threshold)[0]
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]
        
        return torch.tensor(pred_label).to(self.device)

