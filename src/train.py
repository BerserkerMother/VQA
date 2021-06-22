import torch
import torch.nn.functional as F
from torch.utils import data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import os
import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

from data import VQA, get_collate_fn, get_glove_embeddings
from model import New_Net
from utils import AverageMeter, correct, get_current_lr


def main(args):
    # result purposes
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    with open(args.save + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    results = {'loss/train': [], 'loss/test': [], 'lr': [], 'acc/train': [], 'acc/test': []}

    # tensorboard logging
    if not os.path.exists(args.save + '/tensorboard'):
        os.mkdir(args.save + '/tensorboard')
    writer = SummaryWriter(log_dir=args.save + '/tensorboard')

    # create dataset & data loader
    dataset = VQA(args.data)
    args.num_classes = len(dataset.answer2index)
    pad_value = dataset.vocab_stoi['<pad>']
    len_data = dataset.__len__()
    train_samples = int(len_data * .8)
    train_set, test_set = data.random_split(dataset, [train_samples, len_data - train_samples])
    collate_fn = get_collate_fn(pad_value)
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    # load glove embeddings
    if os.path.exists('%s.pth' % args.glove):
        glove_embeddings = torch.load('%s.pth' % args.glove)
    else:
        glove_embeddings = get_glove_embeddings(args.glove, train_set.vocab, args.glove)

    model = New_Net(num_classes=args.num_classes, d_model=args.d_model, dropout=args.dropout,
                    word_embedding=glove_embeddings, num_layers=args.num_layers, num_heads=args.num_heads).cuda()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nmodel has %dM parameters' % (num_parameters // 1000000))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # lambda function for scheduler
    def schedule_fn(x):
        if x > 9:
            return 1 / 5
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_fn)
    scaler = amp.GradScaler()

    epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            results = checkpoint['results']
            epoch = checkpoint['epoch'] + 1

    for e in range(epoch, args.epochs):
        train_loss, train_acc = train(model, optimizer, scaler, train_loader, e, args)
        test_loss, test_acc = None, None
        if args.eval:
            test_loss, test_acc = val(model, test_loader)

        # append results
        results['loss/train'].append(train_loss)
        results['acc/train'].append(train_acc)
        results['loss/test'].append(test_loss)
        results['acc/test'].append(test_acc)
        results['lr'].append(get_current_lr(optimizer))

        # tensorboard logging
        writer.add_scalar('loss/train', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar('acc/train', scalar_value=train_acc, global_step=epoch)
        writer.add_scalar('loss/test', scalar_value=test_loss, global_step=epoch)
        writer.add_scalar('acc/test', scalar_value=test_acc, global_step=epoch)
        writer.add_scalar('lr', scalar_value=get_current_lr(optimizer), global_step=epoch)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'results': results,
            'epoch': e
        }
        torch.save(checkpoint, '%s/checkpoint.pth' % args.save)
        pd.DataFrame(results, index=range(1, e + 1)).to_csv('%s/log.csv' % args.save, index_label='epoch')
        scheduler.step()


def train(model, optimizer, scaler, train_loader, epoch, args):
    top1 = AverageMeter()
    model.train()
    loss_meter = AverageMeter()
    total_loss = 0.
    for i, (qu, im, label, mask) in enumerate(train_loader):
        batch_size = qu.size()[0]
        qu = qu.cuda()
        im = im.cuda()
        label = label.cuda()
        mask = mask.cuda()

        with amp.autocast():
            output = model(qu, im, mask)
            loss = F.cross_entropy(output, label)

        results = correct(output, label, topk=(1,))
        top1.update(results['acc1'], batch_size)

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


def val(model, loader):
    model.eval()
    top1 = AverageMeter()
    loss_meter = AverageMeter()

    print('evaluating...')
    for qu, im, label, mask in tqdm(loader):
        batch_size = qu.size()[0]

        qu = qu.cuda()
        im = im.cuda()
        label = label.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            output = model(qu, im, mask)
            loss = F.cross_entropy(output, label)
            loss_meter.update(loss.item())

            results = correct(output, label, topk=(1,))

            top1.update(results['acc1'], batch_size)

    print('top1 acc: %.2f%%' % (top1.avg() * 100))
    return loss_meter.avg(), top1.avg() * 100


# data related
arg_parser = ArgumentParser(description='New Method for visual question answering')
arg_parser.add_argument('--data', type=str, default='', required=True, help='path to data folder')
arg_parser.add_argument('--batch_size', default=128, type=int, help='batch size')
arg_parser.add_argument('--glove', default='', type=str, help='path to glove text file')
# model related
arg_parser.add_argument('--d_model', type=int, default=512, help='hidden size dimension')
arg_parser.add_argument('--num_layers', type=int, default=6, help='number of encoder layers')
arg_parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
arg_parser.add_argument('--dropout', type=float, default=.2, help='dropout probability')
# optimization related
arg_parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
arg_parser.add_argument('--momentum', type=float, default=.9, help='optimizer momentum')
arg_parser.add_argument('--weight_decay', type=float, default=5e-4, help='optimizer weight decay')
# training related
arg_parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
arg_parser.add_argument('--eval', type=bool, default=True, help='if True evaluates model after every epoch')
arg_parser.add_argument('--save', type=str, default='./run', help='path to save directory')
arg_parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
arg_parser.add_argument('--log_freq', type=int, default=64, help='frequency of logging')

arguments = arg_parser.parse_args()

main(arguments)
