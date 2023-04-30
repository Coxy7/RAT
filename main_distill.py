import json
import os
import torch
import torch.nn as nn
from torchattacks import FFGSM, PGD, TPGD

from utils.experiman import manager
from data import *
from models import get_model
from trainers import DistillTrainer
from losses.trades import TRADESLoss
from losses.distill import DistillLoss
from utils.optim import get_optim
from utils.attacks import SoftPGD


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
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--load_run_name', type=str)
    parser.add_argument('--load_run_number', type=int)
    parser.add_argument('--teacher_run_name', type=str)
    parser.add_argument('--teacher_run_number', type=str)
    ## ===================== Training =========================
    parser.add_argument('--label_smooth', action='store_true')
    parser.add_argument('--distill_temp', default=30, type=float)
    parser.add_argument('--at_mode', type=str)
    parser.add_argument('--beta_trades', default=1, type=float)
    parser.add_argument('--attack', type=str, choices=['fast', 'pgd', 'trades', 'soft_pgd'])
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
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_bb', type=float)
    parser.add_argument('--lr_schedule', type=str)
    parser.add_argument('--multistep_milestones', type=int, nargs='+')
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
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--load_nown', action='store_true')
    parser.add_argument('--rat_alpha', type=float)


def get_pseudo_labels(model, dataloader, device):
    labels_list = []
    idx_list = []
    predictions_list = []
    for (images, labels), idx in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # predictions = model(images).argmax(1)
            predictions = model(images)
        labels_list.append(labels)
        idx_list.append(idx)
        predictions_list.append(predictions)
    labels = torch.cat(labels_list, 0)
    idx = torch.cat(idx_list, 0)
    predictions = torch.cat(predictions_list, 0)
    # acc = predictions.eq(labels).float().mean().item()
    acc = predictions.argmax(1).eq(labels).float().mean().item()
    pseudo_labels = torch.empty_like(predictions)
    pseudo_labels[idx] = predictions
    return pseudo_labels, acc


def main():
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    manager.setup(opt, third_party_tools=( 'tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

    # Data
    logger.info('==> Preparing data..')
    visda = opt.dataset.startswith('visda')
    dataset = get_dataset(opt.dataset)
    loaders = dataset.get_loader(
        opt.data_dir, opt.batch, opt.num_workers,
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
        variant=opt.arch_variant,
        dim=opt.dim,
    ).to(device)
    
    if opt.pretrained or opt.load_run_name:
        if opt.load_nown:
            nn.utils.remove_weight_norm(model.classifier)
        if opt.load_run_name:
            load_dir = manager.get_checkpoint_dir(
                run_name=opt.load_run_name,
                run_number=opt.load_run_number
            )
            load_path = os.path.join(load_dir, 'ckpt.pt')
            logger.info(f'==> Loading model from {load_path} ..')
            state = torch.load(load_path)['model']
            model.load_state_dict(state)
        else:
            logger.info(f'==> Loading pretrained model from {opt.pretrained} ..')
            model.backbone.load_pretrained(opt.pretrained)
        head_parameters = [
            p for n, p in model.named_parameters() if 'backbone' not in n
        ]
        if opt.freeze:
            parameters = head_parameters
        elif opt.lr_bb is not None:
            parameters = [
                {'params': model.backbone.parameters(), 'lr': opt.lr_bb},
                {'params': head_parameters}
            ]
        else:
            parameters = model.parameters()
    else:
        logger.info(f'==> Will train from scratch')
        parameters = model.parameters()
    if opt.arch_variant == 'shot' and not opt.load_nown:
        nn.utils.remove_weight_norm(model.classifier)

    # Teacher model
    if opt.teacher_run_name:
        run_dir = manager.get_run_dir(run_name=opt.teacher_run_name, run_number=opt.teacher_run_number)
        with open(os.path.join(run_dir, 'args.json'), 'r') as fp:
            opt_t = json.load(fp)
        load_dir = manager.get_checkpoint_dir(run_name=opt.teacher_run_name, run_number=opt.teacher_run_number)
        load_path = os.path.join(load_dir, 'ckpt.pt')
        logger.info(f'==> Loading UDA model from {load_path} ..')
        state = torch.load(load_path)['model']
        teacher = get_model(
            arch=opt_t['arch'],
            num_classes=dataset.num_classes,
            preprocess_fn=dataset.preprocess,
            variant=opt_t['arch_variant'],
            dim=opt_t['dim'],
        ).to(device)
        teacher.load_state_dict(state)
        teacher.eval()
        # pseudo_labels, acc = get_pseudo_labels(teacher, raw_trainloader, device)
        # logger.info(f"==> Pseudo label accuracy: {acc:5.3f}")
    else:
        teacher = None

    # Optimizer
    optimizer, scheduler, schedule_step = get_optim(
        parameters=parameters,
        optimizer_name=opt.optimizer,
        lr=opt.lr,
        schedule=opt.lr_schedule,
        weight_decay=opt.weight_decay,
        num_epochs=opt.epoch,
        iters_per_epoch_train=iters_per_epoch_train,
        cyclic_stepsize=opt.cyclic_step,
        multistep_milestones=opt.multistep_milestones,
        adam_beta=opt.adam_beta,
    )

    # Criterions
    criterions=[]
    criterions.append(DistillLoss(temp=opt.distill_temp))
    if opt.at_mode == 'trades':
        criterions.append(TRADESLoss())

    # Trainer
    if opt.attack:
        attack_eps = eval(opt.attack_eps)
        attack_alpha = eval(opt.attack_alpha)
        attack_steps = opt.attack_steps
        if opt.attack == 'fast':
            attack = FFGSM(model, attack_eps, attack_alpha)
        elif opt.attack == 'pgd':
            attack = PGD(model, eps=attack_eps, alpha=attack_alpha,
                         steps=attack_steps, random_start=True)
        elif opt.attack == 'trades':
            attack = TPGD(model, eps=attack_eps, alpha=attack_alpha,
                          steps=attack_steps)
        elif opt.attack == 'soft_pgd':
            attack = SoftPGD(model, teacher, eps=attack_eps, alpha=attack_alpha,
                                 steps=attack_steps, random_start=True)
    else:
        attack = None
    if opt.eval_attack:
        eval_attack_eps = eval(opt.eval_attack_eps)
        eval_attack_alpha = eval(opt.eval_attack_alpha)
        eval_attack_steps = opt.eval_attack_steps
        if opt.eval_attack == 'fast':
            eval_attack = FFGSM(model, eval_attack_eps, eval_attack_alpha)
        elif opt.eval_attack == 'pgd':
            eval_attack = PGD(model, eps=eval_attack_eps, alpha=eval_attack_alpha,
                        steps=eval_attack_steps, random_start=True)
    else:
        eval_attack = None
    trainer = DistillTrainer(
        manager=manager,
        models=(model,),
        dataloaders=(trainloader, raw_trainloader, testloader),
        criterions=criterions,
        optimizers=(optimizer,),
        schedulers=(scheduler,),
        schedule_steps=(schedule_step,),
        num_epochs=opt.epoch,
        iters_per_epoch=(iters_per_epoch_train, iters_per_epoch_test),
        steps_per_iter=(1,),
        accumulation_steps=(opt.accum_steps,),
        device=device,
        attack=attack,
        eval_attack=eval_attack,
        teacher=teacher,
        acc_per_class=visda,
    )

    trainer.train()


if __name__ == "__main__":
    main()
