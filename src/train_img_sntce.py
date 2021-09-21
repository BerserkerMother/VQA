import torch
import torch.nn.functional as F
from torch import sigmoid
from torch.utils import data
from torch.cuda import amp

import wandb
import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

from data import IMSDataset, get_glove_embeddings, get_ism_collate_fn
from model import New_Net
from utils import AverageMeter, get_current_lr, get_sch_fn


# U fucking moron!
def main(args):
    # create dataset & data loader
    train_set = IMSDataset(args.data, splits='train', vocab=None, neg_prob=args.neg_prob)

    val_set = IMSDataset(args.data, splits='val', vocab=train_set.vocab, neg_prob=args.neg_prob)

    pad_value = train_set.word2index['<pad>']
    # split dataset to train and test set
    # gets collate function for data loader
    collate_fn = get_ism_collate_fn(pad_value)
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # load glove embeddings
    if os.path.exists('%s.pth' % args.glove):
        glove_embeddings = torch.load('%s.pth' % args.glove)
    else:
        glove_embeddings = get_glove_embeddings(
            args.glove, train_set.index2word, args.glove)

    model = New_Net(num_classes=1, d_model=args.d_model, attention_dim=args.attention_dim,
                    dropout=args.dropout, word_embedding=glove_embeddings, num_layers=args.num_layers,
                    num_heads=args.num_heads).cuda()

    num_parameters = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
    print('\nmodel has %dM parameters' % (num_parameters // 1000000))

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_sch_fn())
    scaler = amp.GradScaler()

    epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            epoch = checkpoint['epoch'] + 1

    wandb.init(name=args.ex_name, project='VQA',
               entity='berserkermother', config=args)
    for e in range(epoch, args.epochs):
        train_loss, train_acc = train(
            model, optimizer, scaler, train_loader, pad_value, e, args)
        scheduler.step()
        test_loss, test_acc = None, None
        if args.eval:
            test_loss, test_acc = val(
                model, val_loader, pad_value)
        # tensorboard logging
        wandb.log({'loss': {'train': train_loss, 'test': test_loss},
                   'accuracy': {'train': train_acc, 'test': test_acc},
                   'lr': get_current_lr(optimizer)})

        if args.save:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': e
            }
            torch.save(checkpoint, '%s/checkpoint.pth' % args.save)


# fix data loader elements
def train(model, optimizer, scaler, train_loader, pad_value, epoch, args):
    model.train()
    loss_meter, top1, total_loss = AverageMeter(), AverageMeter(), 0.
    for i, batch in enumerate(train_loader):
        caps = batch.caps.cuda(non_blocking=True)
        im_feats = batch.im_feats.cuda(non_blocking=True)
        targets = batch.targets.cuda(non_blocking=True)
        mask = (caps == pad_value)

        batch_size = caps.size()[0]

        with amp.autocast():
            output = model(caps, im_feats, mask=mask)
            loss = F.binary_cross_entropy_with_logits(output, targets, reduction='sum')

        pred = torch.where(output > 0, 1, 0)

        correct = (pred == targets).sum()
        top1.update(correct.item(), batch_size)

        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item())
        total_loss += loss.item()

        if i % args.log_freq == 0 and i != 0:
            print('epoch [%3d/%3d][%4d/%4d], loss: %f, lr: %f' % (
                epoch, args.epochs, i, train_loader.__len__(), total_loss / args.log_freq,
                get_current_lr(optimizer)))
            total_loss = 0.0

    return loss_meter.avg(), top1.avg() * 100


def val(model, loader, pad_value):
    model.eval()

    loss_meter, acc = AverageMeter(), AverageMeter()
    print('evaluating...')
    for batch in tqdm(loader):
        caps = batch.caps.cuda(non_blocking=True)
        im_feats = batch.im_feats.cuda(non_blocking=True)
        targets = batch.targets.cuda(non_blocking=True)
        mask = (caps == pad_value)

        batch_size = caps.size()[0]

        with torch.no_grad():
            output = model(caps, im_feats, mask=mask)
            loss = F.binary_cross_entropy_with_logits(output, targets, reduction='sum')
            loss_meter.update(loss.item())

            pred = torch.where(output > 0, 1, 0)

            correct = (pred == targets).sum()

        acc.update(correct.item(), batch_size)

    print('acc: %f' % acc.avg())

    return loss_meter.avg(), acc.avg()


# data related
arg_parser = ArgumentParser(
    description='trains current architecture on Image-Sentence matching task')
arg_parser.add_argument('--data', type=str, default='',
                        required=True, help='path to data folder')
arg_parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
arg_parser.add_argument('--glove', default='', type=str,
                        help='path to glove text file')
arg_parser.add_argument('--candidate_ans', default='',
                        type=str, help='path to candidate answer json file')
arg_parser.add_argument('--ans_freq', type=int, default=5)
arg_parser.add_argument('--qu_max', type=int, default=14,
                        help='maximum question length')
arg_parser.add_argument('--num_workers', type=int,
                        default=2, help='number of worker to load the data')
arg_parser.add_argument('--neg_prob', type=float,
                        default=.5, help='probability of a sample pairing with negative caption')
# model related
arg_parser.add_argument('--d_model', type=int,
                        default=256, help='hidden size dimension')
arg_parser.add_argument('--attention_dim', type=int, default=256,
                        help='attention dimension for multi head attention')
arg_parser.add_argument('--num_layers', type=int,
                        default=6, help='number of encoder layers')
arg_parser.add_argument('--num_heads', type=int, default=4,
                        help='number of attention heads')
arg_parser.add_argument('--dropout', type=float,
                        default=.2, help='dropout probability')
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4,
                        help='optimizer learning rate')
arg_parser.add_argument('--momentum', type=float,
                        default=.9, help='optimizer momentum')
arg_parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='optimizer weight decay')
# training related
arg_parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
arg_parser.add_argument('--eval', type=bool, default=True,
                        help='if True evaluates model after every epoch')
arg_parser.add_argument('--save', type=str, default='',
                        help='path to save directory')
arg_parser.add_argument('--ex_name', type=str,
                        default='', help='experiment name')
arg_parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint')
arg_parser.add_argument('--log_freq', type=int,
                        default=128, help='frequency of logging')

arguments = arg_parser.parse_args()

arguments.ex_name = ('%s|b%s|d%d|ad%d|l%d|h%d|do%f|lr%f|mo%s|wd%s' % (arguments.ex_name, arguments.batch_size,
                                                                      arguments.d_model, arguments.attention_dim,
                                                                      arguments.num_layers, arguments.num_heads,
                                                                      arguments.dropout, arguments.lr,
                                                                      arguments.momentum, arguments.weight_decay))

main(arguments)
