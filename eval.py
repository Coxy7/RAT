import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn

from torchattacks import FFGSM, PGD, CW, DeepFool, FAB, APGD
from autoattack import AutoAttack
from utils.experiman import manager
from data import *
from models import get_model
from utils.misc import AverageMeter, PerClassMeter
from utils.misc import consume_prefix_in_state_dict_if_present


def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--dataset', type=str, nargs='+')
    parser.add_argument('--train_split', type=str)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('--data_split_seed', default=0, type=int)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    ## ======================= Model ==========================
    parser.add_argument('--load', type=str)
    parser.add_argument('--load_run_name', type=str, nargs='+')
    parser.add_argument('--load_run_number', type=str, nargs='+')
    parser.add_argument('--load_epoch', type=int, nargs='+')
    ## ===================== Training =========================
    parser.add_argument('--attack', type=str)
    parser.add_argument('--attack_eps', type=str)
    parser.add_argument('--attack_alpha', type=str)
    parser.add_argument('--attack_steps', type=int)
    ## ==================== Optimization ======================
    ## ====================== Logging =========================
    parser.add_argument('--comment', default='', type=str)
    ## ==================== Experimental ======================


def eval_acc(model, dataloader, attack, device, is_aa=False, visda=False):

    def update(meter, predictions, labels):
        if visda:
            meter.update(predictions.eq(labels), labels)
        else:
            meter.update(predictions.eq(labels).sum(), labels.size(0))

    if visda:
        print("Using per-class meters for VisDA.")
        meter = PerClassMeter(12, AverageMeter)
        meter_adv = PerClassMeter(12, AverageMeter)
    else:
        meter = AverageMeter()
        meter_adv = AverageMeter()
    bar = tqdm(dataloader)
    for images, labels in bar:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(images).argmax(1)
        update(meter, predictions, labels)
        if attack:
            if is_aa:
                images_adv = attack.run_standard_evaluation(images, labels, bs=images.size(0))
            else:
                images_adv = attack(images, labels)
            with torch.no_grad():
                predictions = model(images_adv).argmax(1)
            update(meter_adv, predictions, labels)
        bar.set_description(f"clean {meter.get_value():5.3f}|adv {meter_adv.get_value():5.3f}")
    acc_clean = meter.get_value() * 100
    acc_adv = meter_adv.get_value() * 100
    return acc_clean, acc_adv


def main():
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    manager.setup(opt)
    logger = manager.get_logger()
    device = 'cuda'

    eps = eval(opt.attack_eps)
    alpha = eval(opt.attack_alpha)
    steps = opt.attack_steps

    for idx, run_name in enumerate(opt.load_run_name):
        dataset = get_dataset(opt.dataset[idx])
        visda = opt.dataset[idx].startswith('visda')
        dataloaders = dataset.get_loader(
            opt.data_dir, opt.batch, opt.num_workers, with_index=False,
            train_split=opt.train_split, val_split=opt.val_split,
            split_seed=opt.data_split_seed)
        testloader = dataloaders[-1]

        run_number = opt.load_run_number[idx] if opt.load_run_number else None
        run_epoch = opt.load_epoch[idx] if opt.load_epoch else None
        run_dir = manager.get_run_dir(run_name, run_number)
        if not os.path.exists(run_dir):
            logger.info(f'Path not exists: {run_dir}')
            continue
        name = f"[{run_name}]"
        if run_number:
            name += f"_n{run_number}"
        if run_epoch:
            name += f"_ep{run_epoch}"

        # load model
        with open(os.path.join(run_dir, 'args.json'), 'r') as fp:
            opt_run = json.load(fp)
        ckpt_name = 'ckpt.pt' if run_epoch is None else f'ckpt-{run_epoch}.pt'
        ckpt_path = os.path.join(manager.get_checkpoint_dir(run_name, run_number), ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.info(f'Path not exists: {ckpt_path}')
            continue
        logger.info(f'==> Building model from {ckpt_path}')
        state = torch.load(ckpt_path)['model']
        model = get_model(
            arch=opt_run['arch'],
            num_classes=dataset.num_classes,
            preprocess_fn=dataset.preprocess,
            variant=opt_run['arch_variant'],
            dim=opt_run['dim'],
        ).to(device)
        consume_prefix_in_state_dict_if_present(state, 'module.')
        if 'classifier.weight_g' not in state:
            nn.utils.remove_weight_norm(model.classifier)
        model.load_state_dict(state)
        model.eval()

        # get attack
        is_aa = (opt.attack == 'autoattack')
        if opt.attack == 'fast':
            attack = FFGSM(model, eps, alpha)
        elif opt.attack == 'pgd':
            attack = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        elif opt.attack == 'autoattack':
            if opt.dataset.startswith('pacs'):
                attack = AutoAttack(model, norm='Linf', eps=eps, version='standard', custom_class_num=6, device=device)
            else:
                attack = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device)
            # attack = AutoAttack(model, norm='Linf', eps=eps, version='rand')
        elif opt.attack == 'cw':
            attack = CW(model, c=3.0 , kappa=0, steps=200)
        elif opt.attack == 'deepfool':
            attack = DeepFool(model, steps=50)
        elif opt.attack == 'fab_l2':
            attack = FAB(model, norm='L2', eps=3.0, steps=100)
        elif opt.attack == 'fab_l1':
            attack = FAB(model, norm='L1', eps=5.0, steps=100)
        elif opt.attack == 'apgd_l2':
            attack = APGD(model, norm='L2', eps=3.0, steps=100)
        else:
            raise NotImplementedError
        print("Attack:", attack)

        # evaluate and record results
        # logger.info(f"Evaluating on training set...")
        # acc_train_clean, acc_train_adv = eval_acc(model, trainloader, attack, device, is_aa)
        # logger.info(f"==> Train acc: clean {acc_train_clean:5.1f}  adv {acc_train_adv:5.1f}")

        # logger.info(f"Evaluating on test set...")
        acc_test_clean, acc_test_adv = eval_acc(
            model, testloader, attack, device, is_aa, visda)
        logger.info(f"==> Test acc:  clean {acc_test_clean:5.1f}  adv {acc_test_adv:5.1f}")

if __name__ == "__main__":
    main()
