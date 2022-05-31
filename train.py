import _init_paths
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model.net import get_model
from dataloader.triplet_img_loader import get_loader
from utils.gen_utils import make_dir_if_not_exist
from utils.vis_utils import vis_with_paths, vis_with_paths_and_bboxes

from config.base_config import cfg, cfg_from_file
from torchvision.utils import save_image
from datetime import datetime

from torchvision import transforms

def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    exp_dir = os.path.join(args.result_dir, args.exp_name)
    make_dir_if_not_exist(exp_dir)

    # Build Model
    model = get_model(args, device)


    if model is None:
        return

    # Criterion and Optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(params, lr=args.lr)

    # Train Test Loop
    init_accuracy_non_margin = 0
    init_accuracy_half_margin = 0
    init_accuracy_quarter_margin = 0

    for epoch in range(1, args.epochs + 1):
        # Init data loaders
        train_data_loader, test_data_loader = get_loader(args)
        # Test train
        train(train_data_loader, model, criterion, optimizer, epoch)
        accuracies = test(test_data_loader, model, criterion)
        # Save model
        model_to_save = {
            "epoch": epoch + 1,
            'state_dict': model.state_dict(),
            "accuracy": [i/len(test_data_loader) for i in accuracies]
        }
        if epoch % args.ckp_freq == 0:
            file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
            save_checkpoint(model_to_save, file_name)

        init_accuracy_non_margin = save_best(init_accuracy_non_margin, accuracies[0], exp_dir, "none", model_to_save)
        init_accuracy_quarter_margin = save_best(init_accuracy_quarter_margin, accuracies[1], exp_dir, "quarter", model_to_save)
        init_accuracy_half_margin = save_best(init_accuracy_half_margin, accuracies[2], exp_dir, "half", model_to_save)


def train(data, model, criterion, optimizer, epoch):
    print("******** Training ********")
    total_loss = 0
    model.train()
    for batch_idx, img_triplet in enumerate(data):
        anchor_img, pos_img, neg_img = img_triplet
        # print("anchor_img", anchor_img.shape)
        if epoch==-1:
            i = 0
            for images in [anchor_img, pos_img, neg_img]:
                for image in images:
                    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

                    inv_tensor = invTrans(image)
                    now = str(datetime.now())
                    path_out = os.path.join("/mnt/28857F714F734EE8/quan_tran/palmline/pytorch-siamese-triplet/augument", now + ".jpg")
                    save_image(inv_tensor, path_out)
                    if i>1000:
                        break
                    i +=1
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_step = args.train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
            total_loss = 0
    print("****************")


def test(data, model, criterion):
    print("******** Testing ********")
    with torch.no_grad():
        model.eval()
        accuracies = [0, 0, 0]
        acc_threshes = [0, 0.25, 0.5]
        total_loss = 0
        for batch_idx, img_triplet in enumerate(data):
            anchor_img, pos_img, neg_img = img_triplet
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
            anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
            E1, E2, E3 = model(anchor_img, pos_img, neg_img)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)

            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            total_loss += loss

            for i in range(len(accuracies)):
                prediction = (dist_E1_E3 - dist_E1_E2 - args.margin * acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc
        print('Test Loss: {}'.format(total_loss / len(data)))
        for i in range(len(accuracies)):
            # if i!=1:
            #     continue
            print(
                'Test Accuracy with diff = {}% of margin: {}'.format(acc_threshes[i] * 100, accuracies[i] / len(data)))
            # print(
            #     'Test Accuracy: {}'.format(accuracies[i] / len(data)))
    print("****************")
    return accuracies

def save_checkpoint(state, file_name):
    torch.save(state, file_name)

def save_best(best_acc, cur_acc, exp_dir, percent, model_to_save):
    # for percent, best_acc, cur_acc in zip(["none", "quarter", "half"], [init_accuracy_non_margin, init_accuracy_quarter_margin, init_accuracy_half_margin],accuracies):
    if best_acc < cur_acc:
        file_name = os.path.join(exp_dir, "best_" + str(percent) + ".pth")
        best_acc = cur_acc
        save_checkpoint(model_to_save, file_name)
    return best_acc


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']      

    # optimizer = torch.optim.SGD(policies,
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                        help="List of GPU Devices to train on")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--ckp_freq', type=int, default=1, metavar='N',
                        help='Checkpoint Frequency (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--margin', type=float, default=5, metavar='M',
                        help='margin for triplet loss (default: 1.0)')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
                        help='Dataset (default: mnist)')

    parser.add_argument('--num_train_samples', type=int, default=50000, metavar='M',
                        help='number of training samples (default: 3000)')
    parser.add_argument('--num_test_samples', type=int, default=10000, metavar='M',
                        help='number of test samples (default: 1000)')

    parser.add_argument('--train_log_step', type=int, default=100, metavar='M',
                        help='Number of iterations after which to log the loss')

    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()
