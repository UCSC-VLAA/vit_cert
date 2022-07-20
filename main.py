import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.recovery import collate_fn_mask

import models_vit

from engine import train_one_epoch, evaluate, cert_recovery

import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finening for certification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=4, type=int)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--name', default='exp name', type=str)

    # Model parameter
    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--lr_schedule', type=str, default='fixed')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameter
    parser.add_argument('--data_path', default='/mnt/disks/sdb_home/ImageNet/ImageNet', type=str)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--output_dir', default='./output_dir/finetune/small400', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--num_cores', default=8, type=int)

    # certified parameters
    parser.add_argument('--cert', default='', type=str)
    parser.add_argument('--width', default=19, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    if args.use_wandb:
        if misc.is_main_process():
            wandb.init(project='smooth-vit-img', config=args, entity='ljb', name=args.name)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    collate_fn = None
    collate_fn = collate_fn_mask(args.width, patch_size=16)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn = collate_fn,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval and not args.cert:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=True)
    
    elif args.cert:
        checkpoint = torch.load(args.cert, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.cert)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        model.load_state_dict(checkpoint_model)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('numbr of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    # certified detection
    if args.cert:
        for mask_num in range(2, 9):
            test_acc1, test_loss, test_acc5, cert = evaluate(data_loader_val, model, epoch=0, device=device, certify='one_quadratic', mask_num=mask_num, num_patches=196)
            if misc.is_main_process():
                with open('cert_results.txt', 'a') as f:
                    f.write("{resume} under {mask_num} attack size: acc: {acc}, cert: {cert} \n".format(resume=args.name, mask_num=mask_num, acc=test_acc1, cert=cert))

        for mask_num in range(2, 9):
            test_acc1, test_loss, test_acc5, cert = evaluate(data_loader_val, model, epoch=0, device=device, certify='one_linear', mask_num=mask_num, num_patches=196)
            if misc.is_main_process():
                with open('cert_results.txt', 'a') as f:
                    f.write("{resume} (linear) under {mask_num} attack size: acc: {acc}, cert: {cert} \n".format(resume=args.name, mask_num=mask_num, acc=test_acc1, cert=cert))

        for mask_num in range(9, 14):
            test_acc1, test_loss, test_acc5, cert = evaluate(data_loader_val, model, epoch = 0, device=device, certify='dual', mask_num=mask_num, num_patches=196)
            if misc.is_main_process():
                with open('cert_results.txt', 'a') as f:
                    f.write("{resume} under two {mask_num} attack size: acc: {acc}, cert: {cert} \n".format(resume=args.name, mask_num=mask_num-7, acc=test_acc1, cert=cert))

        exit(0)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_loss, train_lr = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        if args.output_dir and (epoch % 3 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )

        test_acc1, test_loss, test_acc5 = evaluate(data_loader_val, model, epoch, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_acc1:.1f}%")
        max_accuracy = max(max_accuracy, test_acc1)
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {'epoch': epoch, 'train_loss': train_loss, 'train_lr': train_lr, 'test_acc1': test_acc1, 'test_loss': test_loss, 'test_acc5': test_acc5, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            with open(os.path.join(args.output_dir, "log_array.txt"), mode="a") as f:
                f.write("{:.6g} {:.6g} {:.6g} {:.8g} \n".format(train_loss, test_acc1, test_loss, train_lr))
        
        if args.use_wandb:
            if misc.is_main_process():
                wandb.log({"train_loss": train_loss, "test_acc1": test_acc1, "test_loss": test_loss, "lr": train_lr})

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    for stride in [10,5,1]:
        args.stride = stride
        clean_acc1, smooth_acc, true_avg, certs, true_next_gap = cert_recovery(data_loader_val, model, device, args)

        for i in range(3):
            if misc.is_main_process():
                with open('cert_rec.txt', 'a') as f:
                    f.write(args.name +": clean acc: {clean_acc:.4f}, smooth acc: {smooth_acc:.4f}, cert: {cert:.4f}, true label avg: {true_avg:.4f}, true next gap: {true_next_gap}\
                        width: {width}, stride: {stride}, adv patch size: {adv_size} \n".format(clean_acc=clean_acc1, smooth_acc=smooth_acc, 
                        cert=certs[i], true_avg=true_avg, true_next_gap=true_next_gap, width=args.width, stride=args.stride, adv_size=int(24+8*i)))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)