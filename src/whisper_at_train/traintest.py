# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from loss import Lt_loss

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(3407)

def train(audio_model, train_loader, test_loader, args):
    '''
    wandb.init(
        project=args.model,
    config = {
        'model':args.model,
        'lr':args.lr,
        'lrscheduler_start':args.lrscheduler_start,
        'lrscheduler_step':args.lrscheduler_step,
        'lrscheduler_decay':args.lrscheduler_decay,
        'tdim':args.tdim,
        'epoch':args.n_epochs,
        'mixup':args.mixup,

    },
    group=args.fold
    )
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))


    if args.lr_adapt == True: #作者原本设置的都为False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    elif args.CosineAnnealingLR == True:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.CosTmax, eta_min=args.Cosmin, last_epoch=-1,verbose=True)
        print('----使用余弦退火-----')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'LT':
        loss_fn = nn.CrossEntropyLoss()
        lt_loss = Lt_loss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(args.loss), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 6])
    audio_model.train()
    A_predictions = []
    A_targets = []
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, labels) in enumerate(train_loader):

            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            #if global_step <= 1000 and global_step % 50 == 0 and args.warmup == True:
            #    warm_lr = (global_step / 1000) * args.lr
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = warm_lr
            #    print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output = audio_model(a_input)
                if args.loss =='LT':
                    audio_output,embeddings,_audio_output = audio_output[0],audio_output[1],audio_output[2]
                    loss = loss_fn(audio_output, labels)
                    lt = lt_loss(_audio_output,embeddings)
                    loss = loss + 0.00001*lt
                if args.loss == 'CE':
                    loss = loss_fn(audio_output, labels)

                predictions = audio_output.to('cpu').detach()
                labels = labels.to('cpu').detach()
                A_predictions.append(predictions) 
                A_targets.append(labels)

            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)

            #train_stats = calculate_stats(audio_output, target)

            # optimiztion if amp is used 混合精度
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

            # for audioset-full, break every 10% of the epoch, i.e., equivalent epochs = 0.1 * specified epochs
            if args.dataset == 'as-full':
                if i > 0.1 * len(train_loader):
                    break

        print('start validation')

        stats, valid_loss = validate(audio_model, test_loader, args)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        validate_acc = stats[0]['acc']
        #train_acc = train_stats[0]['acc']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            #print("train_acc: {:.6f}".format(train_acc))
            print("validate_acc: {:.6f}".format(validate_acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        #wandb.log({'acc':acc,'train_loss':loss_meter.avg,'valid_loss':valid_loss})

        result[epoch-1, :] = [validate_acc, loss_meter.avg,valid_loss,mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if validate_acc > best_acc:
            best_acc = validate_acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            pass
            #torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(validate_acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1:.3e}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()
    #wandb.log({'best_acc':best_acc})
def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (a_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device, non_blocking=True)

            with autocast():
                audio_output = audio_model(a_input)
                if args.loss == 'LT':
                    audio_output,embeddings,_audio_output = audio_output[0],audio_output[1],audio_output[2]
                
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)
    return stats, loss