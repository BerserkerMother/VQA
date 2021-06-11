import torch
import torch.nn.functional as F
from torch.utils import data

import os
from argparse import ArgumentParser

from data import VQA, get_collate_fn, get_glove_embeddings
from model import New_Net


def main(args):
    train_set = VQA(args.data)
    pad_value = train_set.vocab.stoi['<pad>']
    collate_fn = get_collate_fn(pad_value)
    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # load glove embeddings
    glove_embeddings = get_glove_embeddings(args.glove, train_set.vocab)

    model = New_Net(num_classes=args.num_classes, d_model=args.d_model, dropout=args.dropout,
                    word_embedding=glove_embeddings, num_layers=args.num_layers, num_heads=args.num_heads)

    optimizer = torch.optim.SGD(params=model.paramaters(), lr=args.lr)

    epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1

    for e in range(epoch, args.epochs):
        train(model, optimizer, train_loader)


def train(model, optimizer, train_loader):
    for qu, im, label, mask in train_loader:
        qu = qu.cuda()
        im = im.cuda()
        label = label.cuda()
        mask = mask.cuda()

        output = model(qu, im, mask)

        loss = F.cross_entropy(output, label)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss: %f' % loss.item())


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
arg_parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
