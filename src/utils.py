from torch.utils.tensorboard import SummaryWriter
import pandas as pd


class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.num = 0

    def reset(self):
        self.sum = 0.
        self.num = 0

    def update(self, value, num=1):
        self.sum += value
        self.num += num

    def avg(self):
        return self.sum / self.num


def correct(output, target, topk=(1,)):
    maxk = max(topk)

    pred = output.topk(maxk, largest=True)[1]
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    results = {}

    for k in topk:
        acc = correct[:, :k].view(-1).sum()
        results['acc%d' % k] = acc.item()

    return results


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def csv_to_tensorboard():
    writer = SummaryWriter('tb')
    data_frame = pd.read_csv('log.csv')
    for index, row in data_frame.iterrows():
        writer.add_scalars('acc', {'train': row['acc/train'], 'test': row['acc/test']}, global_step=index)
        writer.add_scalars('loss', {'train': row['loss/train'], 'test': row['loss/test']}, global_step=index)
        writer.add_scalar('lr', scalar_value=row['lr'], global_step=index + 1)

    writer.close()