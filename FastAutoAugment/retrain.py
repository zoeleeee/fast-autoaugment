'''
export PYTHONPATH=$PYTHONPATH:$PWD
python FastAutoAugment/train.py -c confs/flower_cifar100_10.yaml --aug fa_reduced_cifar10 --dataset cifar100 --dataroot ../data --save cifar100_pyramid272_top1_11.74.pth --nb-labels 10 --beg 20
python FastAutoAugment/train.py -c confs/pyramid272_cifar100_2_tl_re.yaml --aug fa_reduced_cifar10 --dataset cifar100 --dataroot ../data --save cifar100_pyramid272_top1_11.74.pth --nb-labels 10 --beg 20 --binary 0
'''
import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

import torchvision

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from FastAutoAugment.common import get_logger
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.lr_scheduler import adjust_learning_rate_resnet
from FastAutoAugment.metrics import accuracy, Accumulator
from FastAutoAugment.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler
import numpy as np

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)

# use to reverse transform to recover image
# invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
#                                 torchvision.transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465],
#                                                      std = [ 1., 1., 1. ]),
#                                ])

def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, nb_labels=1e6):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        # print(torch.max(data).item(), torch.min(data).item())
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if getattr(optimizer, "synchronize", None):
                optimizer.synchronize()     # for horovod
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, min(5, nb_labels)))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, horovod=False, permutated_vec=None, nb_labels=None, classifier_id=None):
    if horovod:
        import horovod.torch as hvd
        hvd.init()
        device = torch.device('cuda', hvd.local_rank())
        torch.cuda.set_device(device)

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, horovod=horovod, permutated_vec=permutated_vec)
    
    # trying how to recover original image
    # for images, labels in testloader_:
    #     img = invTrans(images[0])
    #     torchvision.utils.save_image(img, 'img.png')
    #     # print(torch.max(images).item(), torch.min(images).item())
    #     return
    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset'], nb_labels), data_parallel=(not horovod))
    # print(model)
    # return
    if C.get()['model'] == 'flower':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    is_master = True
    if horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        optimizer._requires_update = set()  # issue : https://github.com/horovod/horovod/issues/1099
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        if hvd.rank() != 0:
            is_master = False
    logger.debug('is_master=%s' % is_master)

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag or not is_master:
        from FastAutoAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                # only for Pyramid cifar100
                weights = {k.replace('module.', 'module.model.'): v for k, v in data[key].items()}
                weights['fc.weight'] = torch.rand_like(model.state_dict()['fc.weight'])
                weights['fc.bias'] = torch.rand_like(model.state_dict()['fc.bias'])
                model.load_state_dict(weights)
            else:
                weights = {k if 'module.model.' in k else k.replace('module.', 'module.model.'): v for k, v in data[key].items()}
                weights['module.model.fc.weight'] = torch.rand_like(model.state_dict()['module.model.fc.weight'])
                weights['module.model.fc.bias'] = torch.rand_like(model.state_dict()['module.model.fc.bias'])
                weights['module.fc.weight'] = torch.rand_like(model.state_dict()['module.fc.weight'])
                weights['module.fc.bias'] = torch.rand_like(model.state_dict()['module.fc.bias'])
                # weights['module.model.bn']
                model.load_state_dict(weights)
            # optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < C.get()['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
        else:
            print('stop and check the pth file')
            return
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0], nb_labels=nb_labels)
        rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1], nb_labels=nb_labels)
        rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2], nb_labels=nb_labels)
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    path = C.get()['model']
    if not os.path.exists(path):
        os.makedirs(path)
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        if horovod:
            trainsampler.set_epoch(epoch)

        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=is_master, scheduler=scheduler, nb_labels=nb_labels)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 5 == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master, nb_labels=nb_labels)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master, nb_labels=nb_labels)

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if is_master and save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, os.path.join(path, save_path.replace('.pth', '_{}_{}_top1_{:.3f}_{:.3f}.pth'.format(classifier_id, epoch, rs['train']['top1'], rs['test']['top1']))))
    torch.save({
        'epoch': epoch,
        'log': {
            'train': rs['train'].get_dict(),
            'valid': rs['valid'].get_dict(),
            'test': rs['test'].get_dict(),
        },
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }, os.path.join(path, save_path.replace('.pth', '_{}_{}_top1_{:.3f}_{:.3f}.pth'.format(classifier_id, epoch, rs['train']['top1'], rs['test']['top1']))))

    del model

    result['top1_test'] = best_top1
    return result

def binary_decimal(permutated_vec, nb_labels):
    std = np.array([np.power(2,i) for i in range(nb_labels)])
    return np.sum(permutated_vec.T*std, axis=-1)


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='test.pth')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--horovod', action='store_true')
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--classifier-id', type=int, default=0)
    parser.add_argument('--nb-labels', type=int, default=1e6)
    parser.add_argument('--beg', type=int, default=0)
    parser.add_argument('--binary', type=int, default=1)
    args = parser.parse_args()

    assert not (args.horovod and args.only_eval), 'can not use horovod when evaluation mode is enabled.'
    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    permutated_vec = None
    nb_labels = 1e6
    if args.nb_labels != 1e6:
        nb_labels = args.nb_labels
        if os.path.exists('{}_label_permutation_cifar100.npy'.format(nb_labels)):
            permutated_vec = np.load('{}_label_permutation_cifar100.npy'.format(nb_labels))[int(args.classifier_id)]
        else:
            idxs = np.load('order.npy')#np.arange(int(args.beg), int(args.beg)+nb_labels)
            permutated_vec = np.load('2_label_permutation_cifar100.npy')[idxs]
            if args.binary == 1:
                permutated_vec = binary_decimal(permutated_vec[args.beg:args.beg+args.nb_labels], nb_labels)
        

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, horovod=args.horovod, metric='test', permutated_vec=permutated_vec, nb_labels=nb_labels, classifier_id=args.classifier_id)
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
