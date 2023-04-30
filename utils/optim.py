import math
import torch.optim as optim


def get_optim(parameters, optimizer_name, lr, schedule,
              weight_decay, num_epochs, iters_per_epoch_train,
              cyclic_stepsize=None, multistep_milestones=None, adam_beta=0.5):

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay,
            nesterov=False)
    elif optimizer_name == 'sgd_nesterov':
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay,
            nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            parameters, lr=lr, weight_decay=weight_decay, betas=(adam_beta, 0.999))
            # parameters, lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay, betas=(adam_beta, 0.999))

    if schedule == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs)
        schedule_step = 'epoch'
    elif schedule == '1cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            epochs=num_epochs, steps_per_epoch=iters_per_epoch_train,
            pct_start=0.25, anneal_strategy='cos')
        schedule_step = 'iter'
    elif schedule == 'cyclic':
        if cyclic_stepsize is None:
            cyclic_stepsize = 0.5 * num_epochs
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0, max_lr=lr,
            cycle_momentum=(optimizer_name == 'sgd'),
            step_size_up=int(cyclic_stepsize * iters_per_epoch_train),
        )
        schedule_step = 'iter'
    elif schedule == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=multistep_milestones, gamma=0.1)
        schedule_step = 'epoch'
    elif schedule == 'dao':     # used in SRDC, SHOT
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            (lambda epoch: math.pow((1 + 10 * epoch / num_epochs), -0.75))
        )
        schedule_step = 'epoch'
    elif schedule == 'none':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, (lambda epoch: 1))
        schedule_step = 'epoch'
    
    return optimizer, scheduler, schedule_step
