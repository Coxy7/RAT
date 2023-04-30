import os
import torch
import torch.nn as nn
from torchattacks import FFGSM, PGD, TPGD

from utils.experiman import manager
from data import *
from models import get_model
from trainers import SHOTTrainer
from losses.shot import IMLoss
from losses.trades import TRADESLoss
from utils.optim import get_optim


def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_split', type=str)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('--data_split_seed', default=0, type=int)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    ## ======================= Model ==========================
    parser.add_argument('--arch', type=str)
    parser.add_argument('--arch_variant', type=str, default='std')
    parser.add_argument('--dim', type=int)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--load_run_name', type=str)
    parser.add_argument('--load_run_number', type=str)
    parser.add_argument('--teacher_run_name', type=str)
    parser.add_argument('--teacher_run_number', type=str)
    ## ===================== Training =========================
    parser.add_argument('--lambda_cls', type=float, default=0.3)
    parser.add_argument('--lambda_im', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--at_mode', type=str, choices=['pgd_at', 'trades'])
    parser.add_argument('--beta_trades', default=1.0, type=float)
    parser.add_argument('--attack', type=str, choices=['pgd', 'trades'])
    parser.add_argument('--attack_eps', type=str)
    parser.add_argument('--attack_alpha', type=str)
    parser.add_argument('--attack_steps', type=int)
    parser.add_argument('--eval_attack', type=str, choices=['fast', 'pgd'])
    parser.add_argument('--eval_attack_eps', type=str)
    parser.add_argument('--eval_attack_alpha', type=str)
    parser.add_argument('--eval_attack_steps', type=int)
    ## ==================== Optimization ======================
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--iters_per_epoch_train', type=int,
                        help="default: len(trainloader)")
    parser.add_argument('--iters_per_epoch_test', type=int,
                        help="default: len(testloader)")
    parser.add_argument('--steps_per_iter', default=1, type=int)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_bb', type=float)
    parser.add_argument('--lr_schedule', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--adam_beta', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--cyclic_step', type=float)
    ## ====================== Logging =========================
    parser.add_argument('--log_clean_acc', action='store_true')
    parser.add_argument('--log_period', default=5, type=int, metavar='LP',
                        help='log every LP iterations')
    parser.add_argument('--ckpt_period', default=10, type=int, metavar='CP',
                        help='make checkpoints every CP epochs')
    parser.add_argument('--test_period', default=1, type=int, metavar='TP',
                        help='test every TP epochs')
    parser.add_argument('--comment', default='', type=str)
    ## ==================== Experimental ======================


def get_pseudo_labels(model, dataloader, device):
    labels_list = []
    idx_list = []
    predictions_list = []
    for (images, labels), idx in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(images).argmax(1)
        labels_list.append(labels)
        idx_list.append(idx)
        predictions_list.append(predictions)
    labels = torch.cat(labels_list, 0)
    idx = torch.cat(idx_list, 0)
    predictions = torch.cat(predictions_list, 0)
    acc = predictions.eq(labels).float().mean().item()
    pseudo_labels = torch.empty_like(labels)
    pseudo_labels[idx] = predictions
    return pseudo_labels, acc


def main():
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    manager.setup(opt, third_party_tools=('tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

    # Data
    logger.info('==> Preparing data..')
    visda = opt.dataset.startswith('visda')
    dataset = get_dataset(opt.dataset)
    loaders = dataset.get_loader(
        opt.data_dir, opt.batch, opt.num_workers, with_index=True,
        train_split=opt.train_split,
        val_split=opt.val_split,
        split_seed=opt.data_split_seed)
    if opt.val_split:
        trainloader, raw_trainloader, valloader, testloader = loaders
        testloader = valloader
    else:
        trainloader, raw_trainloader, testloader = loaders
    if opt.iters_per_epoch_train:
        iters_per_epoch_train = opt.iters_per_epoch_train
    else:
        iters_per_epoch_train = len(trainloader) // opt.accum_steps
    if opt.iters_per_epoch_test:
        iters_per_epoch_test = opt.iters_per_epoch_test
    else:
        iters_per_epoch_test = len(testloader)
    logger.info(f"Iters per epoch: train {iters_per_epoch_train}, test {iters_per_epoch_test}")

    # Model
    logger.info('==> Building models..')
    model = get_model(
        arch=opt.arch,
        num_classes=dataset.num_classes,
        preprocess_fn=dataset.preprocess,
        variant='shot',
        dim=opt.dim,
    ).to(device)

    if opt.load or opt.load_run_name:
        if opt.load:
            load_path = opt.load
        else:
            load_dir = manager.get_checkpoint_dir(run_name=opt.load_run_name, run_number=opt.load_run_number)
            load_path = os.path.join(load_dir, 'ckpt.pt')
        logger.info(f'==> Loading source model from {load_path} ..')
        state = torch.load(load_path)['model']
        model.load_state_dict(state)
    elif opt.pretrained:
        logger.info(f'==> Loading pretrained model from {opt.pretrained} ..')
        model.backbone.load_pretrained(opt.pretrained)
    else:
        logger.info(f'==> Will train from scratch')

    # Pseudo label
    if opt.teacher_run_name:
        load_dir = manager.get_checkpoint_dir(run_name=opt.teacher_run_name, run_number=opt.teacher_run_number)
        load_path = os.path.join(load_dir, 'ckpt.pt')
        logger.info(f'==> Loading UDA model from {load_path} ..')
        state = torch.load(load_path)['model']
        teacher = get_model(
            arch=opt.arch,
            num_classes=dataset.num_classes,
            preprocess_fn=dataset.preprocess,
            variant='shot',
            dim=opt.dim,
        ).to(device)
        teacher.load_state_dict(state)
        teacher.eval()
        pseudo_labels, acc = get_pseudo_labels(teacher, raw_trainloader, device)
        logger.info(f"==> Pseudo label accuracy: {acc:5.3f}")
    else:
        pseudo_labels = None

    # Optimizer
    lr_backbone = opt.lr * 0.1 if opt.lr_bb is None else opt.lr_bb
    parameters = [
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.bottleneck.parameters()}
    ]
    optimizer, scheduler, schedule_step = get_optim(
        parameters=parameters,
        optimizer_name=opt.optimizer,
        lr=opt.lr,
        schedule=opt.lr_schedule,
        weight_decay=opt.weight_decay,
        num_epochs=opt.epoch,
        iters_per_epoch_train=iters_per_epoch_train,
        cyclic_stepsize=opt.cyclic_step,
        adam_beta=opt.adam_beta,
    )

    # Criterions
    criterions=[
        nn.CrossEntropyLoss(),
        IMLoss(),
    ]
    if opt.at_mode == 'trades':
        criterions.append(TRADESLoss())

    # Trainer
    if opt.attack:
        attack_eps = eval(opt.attack_eps)
        attack_alpha = eval(opt.attack_alpha)
        attack_steps = opt.attack_steps
        if opt.attack == 'pgd':
            attack = PGD(model, eps=attack_eps, alpha=attack_alpha,
                         steps=attack_steps, random_start=True)
        elif opt.attack == 'trades':
            attack = TPGD(model, eps=attack_eps, alpha=attack_alpha,
                          steps=attack_steps)
    else:
        attack = None

    if opt.eval_attack:
        eval_attack_eps = eval(opt.eval_attack_eps)
        eval_attack_alpha = eval(opt.eval_attack_alpha)
        eval_attack_steps = opt.eval_attack_steps
        if opt.eval_attack == 'fast':
            eval_attack = FFGSM(model, eval_attack_eps, eval_attack_alpha)
        elif opt.eval_attack == 'pgd':
            eval_attack = PGD(model, eps=eval_attack_eps, alpha=eval_attack_alpha, steps=eval_attack_steps, random_start=True)
    else:
        eval_attack = None

    trainer = SHOTTrainer(
        manager=manager,
        models=(model,),
        dataloaders=(trainloader, raw_trainloader, testloader),
        criterions=criterions,
        optimizers=(optimizer,),
        schedulers=(scheduler,),
        schedule_steps=(schedule_step,),
        num_epochs=opt.epoch,
        iters_per_epoch=(iters_per_epoch_train, iters_per_epoch_test),
        steps_per_iter=(opt.steps_per_iter,),
        accumulation_steps=(opt.accum_steps,),
        device=device,
        attack=attack,
        eval_attack=eval_attack,
        pseudo_labels=pseudo_labels,
        cls_loss_start_epoch=(1 if visda else 0),
        acc_per_class=visda,
    )

    trainer.train()


if __name__ == "__main__":
    main()
