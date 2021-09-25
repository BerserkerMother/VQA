import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

import json
import os
import glob
import random
import numpy as np

from .utils import get_vocab_ism
from .path import IMAGE_SENTENCE_MATCHING


class IMSDataset(Dataset):
    def __init__(self, root, splits, vocab, neg_prob):
        self.root = root
        self.splits = splits.split('+')
        self.neg_prob = neg_prob

        # make data
        self.data = []
        dict_ = {}
        for split in self.splits:
            path = os.path.join(self.root, IMAGE_SENTENCE_MATCHING[split + '_an'])
            annotations = json.load(open(path, 'r'))['annotations']
            for ann in annotations:
                # TODO: add caption preprocessing to remove , . and unnecessary punctuations
                if ann['image_id'] in dict_:
                    dict_[ann['image_id']].append(ann['caption'])
                else:
                    dict_[ann['image_id']] = [ann['caption']]

        for key, value in dict_:
            self.data.append((key, value))

        # get image id image path
        self.im_id2im_path = {}
        for split in self.splits:
            path = os.path.join(self.root, IMAGE_SENTENCE_MATCHING[split + '_im'])
            for im_path in glob.glob(path + '/*'):
                im_id = str(int(im_path.split('/')[-1].split('.')[0].split('_')[-1]))
                self.im_id2im_path[im_id] = im_path

        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        # make vocab
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = get_vocab_ism(self.data, self.tokenizer)

        self.word2index = self.vocab.get_stoi()
        self.index2word = self.vocab.get_itos()

    def __getitem__(self, idx):
        im_id, caps = self.data[idx]

        # load image features
        im_feat = np.load(self.im_id2im_path[im_id])
        im_feat = torch.tensor(im_feat, dtype=torch.float32)

        # possibly use negative caption
        caps, indices = zip(*[self.sample_caption(cap, im_id) for cap in caps])
        caps_tensor = []
        for cap in caps:
            # tokenize caption and make tensor
            cap_ = []
            for word in self.tokenizer(cap):
                if cap in self.word2index:
                    cap_.append(self.word2index[word])
                else:
                    cap_.append(self.word2index['<unk>'])
            cap = torch.tensor(cap_, dtype=torch.long)
            caps_tensor.append(cap)

        return im_feat, caps, indices

    def __len__(self):
        return len(self.data)

    def sample_caption(self, cap, im_id):
        if random.random() < self.neg_prob:
            # get random negative caption
            index = 0
            cap = self.negative_caption(im_id)
            return cap, index
        else:
            index = 1
            return cap, index

    def negative_caption(self, im_id, num_try=5):
        for i in range(num_try):
            idx_ = np.random.choice(np.arange(len(self.data)))
            # if image_ids are different then return the caption
            if im_id != self.data[idx_][0]:
                return self.data[idx_][1]

        raise Warning('Could not find negative sample within %d tries' % num_try)
