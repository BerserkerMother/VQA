import torch
import torch.nn.functional as F
from torch.utils import data
from torch.cuda import amp

import os
import json
import datetime
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
    date = datetime.datetime.now().__str__()
    path = os.path.join(args.save, date)
    os.mkdir(path)
    with open(path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)
    results = {'epoch': [], 'loss': [], 'lr': [], 'acc@1': [], 'acc@5': []}
    args.save = path

    # create data loader
    train_set = VQA(args.data)
    pad_value = train_set.vocab_stoi['<pad>']
    collate_fn = get_collate_fn(pad_value)
    train_loader = data.DataLoader(
        dataset=train_set,
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

    model = New_Net(num_classes=len(train_set.answer2index), d_model=args.d_model, dropout=args.dropout,
                    word_embedding=glove_embeddings, num_layers=args.num_layers, num_heads=args.num_heads).cuda()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nmodel has %dM parameters' % (num_parameters // 1000000))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    scaler = amp.GradScaler()

    epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            results = checkpoint['results']
            epoch = checkpoint['epoch'] + 1

    for e in range(epoch, args.epochs):
        loss = train(model, optimizer, scaler, train_loader, e, args)
        acc1, acc5 = val(model, train_loader)

        # append results
        results['epoch'].append(e)
        results['loss'].append(loss)
        results['acc@1'].append(acc1)
        results['acc@5'].append(acc5)
        results['lr'].append(get_current_lr(optimizer))

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'results': results,
            'epoch': e
        }
        torch.save(checkpoint, '%s/checkpoint.pth' % args.save)
        pd.DataFrame(results, index=range(1, e + 1)).to_csv('%s/log.csv' % args.save, index_label='epoch')


def train(model, optimizer, scaler, train_loader, epoch, args):
    model.train()
    loss_meter = AverageMeter()
    total_loss = 0.
    for i, (qu, im, label, mask) in enumerate(train_loader):
        qu = qu.cuda()
        im = im.cuda()
        label = label.cuda()
        mask = mask.cuda()

        with amp.autocast():
            output = model(qu, im, mask)
            loss = F.cross_entropy(output, label)

        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % 24 == 0 and i != 0:
            print('epoch [%3d/%3d][%3d/%3d], loss: %f, lr: %f' % (
                epoch, args.epochs, i, train_loader.__len__(), total_loss / 24,
                get_current_lr(optimizer)))
            loss_meter.update(total_loss)
            total_loss = 0.0

    return loss_meter.avg()


def val(model, loader):
    model.eval()
    top1, top5 = AverageMeter(), AverageMeter()

    print('evaluating...')
    for qu, im, label, mask in tqdm(loader):
        batch_size = qu.size(0)

        qu = qu.cuda()
        im = im.cuda()
        label = label.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            output = model(qu, im, mask)

            results = correct(output, label, topk=(1, 5))

            top1.update(results['acc1'], batch_size)
            top5.update(results['acc5'], batch_size)

    print('top1 acc: %.2f | top5 acc: %.2f' % (top1.avg() * 100, top5.avg() * 100))
    return results['acc1'], results['acc5']


# data related
arg_parser = ArgumentParser(description='New Method for visual question answering')
arg_parser.add_argument('--data', type=str, default='', required=True, help='path to data folder')
arg_parser.add_argument('--batch_size', default=512, type=int, help='batch size')
arg_parser.add_argument('--glove', default='', type=str, help='path to glove text file')
# model related
arg_parser.add_argument('--d_model', type=int, default=768, help='hidden size dimension')
arg_parser.add_argument('--num_layers', type=int, default=6, help='number of encoder layers')
arg_parser.add_argument('--num_heads', type=int, default=12, help='number of attention heads')
arg_parser.add_argument('--lr', type=float, default=1e-3, help='optimizer learning rate')
arg_parser.add_argument('--dropout', type=float, default=.2, help='dropout probability')
# training related
arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
arg_parser.add_argument('--save', type=str, default='./run', help='path to save directory')
arg_parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')

arguments = arg_parser.parse_args()

main(arguments)
