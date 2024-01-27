import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter


def train_cnn(optimizer, model, train_generator, valid_generator, criterion, args):
    n_iteration_per_epoch = len(train_generator)

    tb_logger = SummaryWriter(log_dir=args.output)
    print(n_iteration_per_epoch)
    step_size = 100  # Apply adjustment every 10 epochs
    lr_initial = 1e-4
    lambda_lr = lambda step: lr_initial * ((step // step_size + 1) ** 0.5) if step > 0 else lr_initial
    scheduler = LambdaLR(optimizer, lambda_lr)
    train_losses = []
    valid_losses_l1 = []
    avg_train_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0
    for epoch in range(args.epoch + 1):
        model.train()

        iteration = 0
        total_step = 0
        for idx, (image, gt, img_upsampled) in enumerate(train_generator):
            # scheduler.step()
            total_step += args.b
            img = image.cuda(non_blocking=True).float()
            gt = gt.cuda(non_blocking=True).float()
            img_upsampled = img_upsampled.cuda(non_blocking=True).float()
            pred = model(img)
            loss = criterion(img_upsampled, pred, gt)
            iteration += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch * n_iteration_per_epoch + iteration)
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 20 == 0:
                print('Epoch [{}/{}], iteration {}, l1oss:{:.6f}, {:.6f} ,learning rate{:.6f}'
                      .format(epoch + 1, args.epoch, iteration + 1, loss.sum().item(), np.average(train_losses),
                              optimizer.state_dict()['param_groups'][0]['lr']))
                sys.stdout.flush()

        with torch.no_grad():
            model.eval()
            print("validating....")
            for i, (image, gt, _) in enumerate(valid_generator):
                image_scale = image.cuda(non_blocking=True).float()
                gt_scale = gt.cuda(non_blocking=True).float()
                pred = model(image_scale)
                loss = criterion(pred, gt_scale)
                valid_losses_l1.append(loss.sum().item())
        # logging
        train_loss = np.average(train_losses)
        valid_loss_l1 = np.average(valid_losses_l1)
        valid_loss = valid_loss_l1
        tb_logger.add_scalar('valid loss', valid_loss_l1, epoch)
        avg_train_losses.append(train_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        train_losses = []
        valid_losses = []

        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            # save model
            # save all the weight for 3d unet
            torch.save({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output,
                            'best' + '.pt'))
            print("Saving model ",
                  os.path.join(args.output,
                               'best' + '.pt'))
        else:
            if epoch % 10 == 0:
                torch.save({
                    'args': args,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.output,
                                'epoch_' + str(epoch) + '.pt'))
                print("Saving model ",
                      os.path.join(args.output,
                                   'epoch_' + str(epoch) + '.pt'))
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement > 10:
                break
        sys.stdout.flush()
    tb_logger.close()
