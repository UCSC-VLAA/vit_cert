import torch
from util import misc
from util import mask
from util import lr_sched
from typing import Iterable

from timm.utils import accuracy
import math
import sys

from util.recovery import rec_mask

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter

        lr = optimizer.param_groups[-1]["lr"]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['lr'].update(lr, n=1)
        
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
    
    metric_logger.synchronize_between_processes()
    print("Averaged status:", metric_logger)

    return metric_logger.loss.global_avg, metric_logger.lr.global_avg



@torch.no_grad()
def evaluate(data_loader, model, epoch, device, certify=None, mask_num=None, num_patches=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1,5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if certify == None:
            continue
        if certify == 'one_quadratic':
            patch_num = int(math.sqrt(num_patches))
            consistency = torch.argmax(output, dim=1).eq(target).float()
            for pos in range(patch_num * (patch_num - mask_num + 1)):
                if pos % patch_num > patch_num - mask_num:
                    continue
                mask_idx = mask.mask_quadratic(pos, mask_num, patch_num ** 2)
                output = model(images, mask_idx)
                consistency_pos = torch.argmax(output, dim=1).eq(target).float()
                if pos % 10 == 0 and misc.is_main_process():
                    print(pos)
                    print(torch.sum(consistency)/batch_size, torch.sum(consistency))
                consistency = torch.mul(consistency, consistency_pos)
        elif certify == 'one_linear':
            patch_num = int(math.sqrt(num_patches))
            consistency = torch.argmax(output, dim=1).eq(target).float()
            for pos in range(patch_num - mask_num + 1):
                mask_idx = mask.mask_linear(pos, mask_num, patch_num**2)
                output = model(images, mask_idx)
                consistency_pos = torch.argmax(output, dim=1).eq(target).float()
                if misc.is_main_process():
                    print(pos)
                    print(torch.sum(consistency)/batch_size, torch.sum(consistency))
                consistency = torch.mul(consistency, consistency_pos)
        elif certify == 'dual':
            patch_num = int(math.sqrt(num_patches))
            consistency = torch.argmax(output, dim=1).eq(target).float().cpu()
            for pos in range(num_patches):
                mask_idx = mask.mask_quadratic(pos, mask_num, num_patches)
                output = model(images, mask_idx)
                consistency_pos = torch.argmax(output, dim=1).eq(target).float().cpu()
                if pos % 10 == 0 and misc.is_main_process():
                    print(pos)
                    print(torch.sum(consistency)/batch_size, torch.sum(consistency))
                consistency = torch.mul(consistency, consistency_pos)
        else:
            print('certify shuold be one_quadratic, one_linear or dual!')

        metric_logger.meters['cert'].update(torch.sum(consistency)/batch_size, n=batch_size)

    # gather
    metric_logger.synchronize_between_processes()
    acc1 = metric_logger.meters['acc1'].global_avg
    acc5 = metric_logger.meters['acc5'].global_avg
    loss = metric_logger.meters['loss'].global_avg
    if certify == None:
        if misc.is_main_process():
            print('* Acc@1 {top1:.3f} Acc@5 {top5:.3f} loss {losses:.3f}'
                .format(top1=acc1, top5=acc5, losses=loss))
        return acc1, acc5, loss
    else:
        cert = metric_logger.meters['cert'].global_avg

        if misc.is_main_process():
            print('* Acc@1 {top1:.3f} Cert {cert:.3f} Acc@5 {top5:.3f} loss {losses:.3f}'
                .format(top1=acc1, cert=cert, top5=acc5, losses=loss))
        return acc1, loss, acc5, cert



@torch.no_grad()
def cert_recovery(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test cert_rec:'

    # switch to evaluation mode
    model.eval()

    gaps = [2 * math.ceil((adv_size + args.width - 1)/args.stride) + 1 for adv_size in [24, 32, 40]]

    total_cert_samples, total_cert_batches, total_clean_acc, total_smooth_acc, total_true_avg, total_true_next_gap = 0, 0, 0.0, 0.0, 0.0, 0.0
    total_certs = [0.0, 0.0, 0.0]

    for data_iter_step, (images, target) in enumerate(data_loader):
        batch_size, c, h, w = images.shape
        images = images.to(device)
        target = target.to(device)
        count = torch.zeros(batch_size, args.nb_classes).to(device)
        
        for pos in range(0, w, args.stride):
            images_ = rec_mask(images, args.width, pos)
            # if pos + args.width < w:
                # output = model(torch.gather(images, dim=-1, index=index), pos=pos)
                # output = model(images)
            # else:
                # index = torch.tensor(list(range(pos, w)) + list(range(0, (pos+args.width)%w))).repeat(batch_size, c, h, 1).to(device)
                # output = model(torch.gather(images, dim=-1, index=index), pos=pos)
            output = model(images_)

            predictions = torch.argmax(output, dim=1)
            count[torch.tensor(range(batch_size)), predictions] += 1

        output = model(images)
        clean_acc1, clean_acc5 = accuracy(output, target, topk=(1,5))


        y_1st_vals, y_1st_idx = count.kthvalue(args.nb_classes, dim=1)
        y_2nd_vals, y_2nd_idx = count.kthvalue(args.nb_classes-1, dim=1)
        
        y_tar_vals = torch.gather(count, 1, target.unsqueeze(1)).squeeze()
        not_y = (y_1st_idx != target)
        # max out of true value (max if correct otherwise 2nd max)
        y_nex_idx = y_1st_idx*(not_y.int()) + y_2nd_idx*(~not_y)
        y_nex_vals = torch.gather(count, 1, y_nex_idx.unsqueeze(1)).squeeze()
        
        y_certified = [(target == y_1st_idx) * (y_1st_vals >= y_2nd_vals + gap) for gap in gaps]

        metric_logger.meters['clean_acc'].update(clean_acc1, n=batch_size)
        metric_logger.meters['smooth_acc'].update(torch.sum(y_1st_idx == target)/batch_size, batch_size)
        metric_logger.meters['true_avg'].update(torch.sum(y_tar_vals)/batch_size, batch_size)
        for i in range(3):
            metric_logger.meters['certs' + str(i)].update(torch.sum(y_certified[i])/batch_size, batch_size)
        metric_logger.meters['true_next_gap'].update(torch.sum(y_tar_vals - y_nex_vals)/batch_size, batch_size)

        # if data_iter_step % 50 == 0:
            # print('Device: ' + str(dist.get_rank()) + ' step: ' + str(data_iter_step) + 'true: ' + str(torch.sum(y_1st_idx == target)) + ' certs 24: ' + str(torch.sum(y_certified[i]).tolist()))

    metric_logger.synchronize_between_processes()
    clean_acc1 = metric_logger.meters['clean_acc'].global_avg
    smooth_acc = metric_logger.meters['smooth_acc'].global_avg
    true_avg = metric_logger.meters['true_avg'].global_avg
    certs = [metric_logger.meters['certs'+str(i)].global_avg for i in range(3)]
    true_next_gap = metric_logger.meters['true_next_gap'].global_avg

    return clean_acc1, smooth_acc, true_avg, certs, true_next_gap
