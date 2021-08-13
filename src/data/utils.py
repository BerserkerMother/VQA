# data utility modules
import torch
from torch.utils.data import Subset
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import json
import random
from collections import Counter
from tqdm import tqdm


def get_answer_dict(path):
    with open(path, 'r') as f:
        dic = json.load(f)[0]

    index2answer = [value for value in dic.values()]

    return dic, index2answer


# takes questions vocab and GloVe text and returns tensor containing weights
def get_glove_embeddings(path, question_vocab, save_path):
    """

    :param path: path to glove embedding text file
    :param question_vocab: pytorch Vocab object contains all in use words
    :param save_path: path to save glove embedding pth fie so U won't have to process again
    :return: tensor of size (vocab_size, embedding_dim)
    """
    with open('%s.txt' % path, 'r') as file:
        text = file.read().split('\n')  # each starts with the character and following it its values

    # extract words and features from text file
    word_to_features = {}
    for line in text:
        line = line.split(' ')
        word_to_features[line[0]] = [float(num) for num in line[1:]]

    # make desired glove embedding
    itos = question_vocab.get_itos()
    glove_embeddings = torch.tensor([])
    for word in tqdm(itos):
        if word in word_to_features.keys():
            temp = torch.tensor([word_to_features[word]])
        else:
            temp = torch.zeros((1, 300))

        glove_embeddings = torch.cat((glove_embeddings, temp))

    torch.save(glove_embeddings, '%s.pth' % save_path)

    return glove_embeddings


# given questions makes vocab
def make_vocab(questions, tokenizer):
    """

    :param questions: dic (question_id, question_text)
    :param tokenizer: english tokenizer
    :return: pytorch Vocab object
    """
    counter = Counter()
    for question in questions.values():
        counter.update(tokenizer(question))

    v = vocab(counter, min_freq=5)
    v.insert_token('<unk>', 0)
    v.insert_token('<pad>', 1)
    v.set_default_index(v['<unk>'])

    return v


def dataset_random_split(dataset, ratio=.8, seed=0):
    """

    :param dataset: dataset object to be divided
    :param ratio: ratio of train samples
    :param seed: random seed
    :return: a train and test set
    """
    len_data = dataset.__len__()
    train_size = int(len_data * ratio)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len_data))
    np.random.shuffle(indices)

    return Subset(dataset, indices[:train_size]), Subset(dataset, indices[train_size:])


class CustomBatch:
    """
    Custom batch class so we can pin memory our custom dataset and loader
    """

    def __init__(self, data, pad_value):
        """

        :param data: list of tuples contain question tensor, img feature tensor and answer label
        :param pad_value:
        """

        # qu_list contains list of tensor with variant size
        # im_list contains list of image feature tensors
        # ans_list contains list of answer index
        qu_list, im_list, im_box, ans_list = zip(*data)

        # pads questions to max question length in current batch
        self.qu = pad_sequence(qu_list, padding_value=pad_value).permute(1, 0)
        self.im = torch.stack(im_list, dim=0)
        self.im_box = torch.stack(im_box, dim=0)
        self.ans = torch.tensor(ans_list, dtype=torch.long)

    # pin memory function for data loader
    def pin_memory(self):
        self.qu.pin_memory()
        self.im.pin_memory()
        self.im_box.pin_memory()
        self.ans.pin_memory()

        return self


def get_collate_fn(pad_value):
    """

    :param pad_value: index of pad token in vocab
    :return: a function for data loader collate_fn
    """

    def collate_fn(data):
        return CustomBatch(data, pad_value)

    return collate_fn
