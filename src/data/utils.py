import torch
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from collections import Counter
from tqdm import tqdm
import json


def get_answer_dict(path):
    with open(path, 'r') as f:
        dic = json.load(f)[0]

    index2answer = [key for key in dic.keys()]

    return dic, index2answer


def get_vocab(questions, tokenizer):
    """

    :param questions: list of questions dictionary
    :param tokenizer: tokenizer for sentence
    :return: torch Vocab object
    """
    counter = Counter()
    for question in questions['questions']:
        counter.update(tokenizer(question['question']))

    v = vocab(counter, min_freq=3)
    v.insert_token('<unk>', 0)
    v.insert_token('<pad>', 1)
    v.set_default_index(v['<unk>'])

    return v


def get_candidate_answers(annotations):
    """

    :param annotations: list of annotations dictionary
    :return: tuple(answer2index, index2answer)
    """
    return NotImplemented


def is_answers_in_bank(answers, candidate):
    for answer in answers:
        if answer in candidate:
            return True
    return False


def get_answers_array(answers):
    return [ans['answer'] for ans in answers]


def get_ans_scores(answers, candidate):
    scores = np.zeros((len(candidate)))

    for answer in answers:
        if answer in candidate:
            scores[candidate[answer]] += 1

    return scores / 3.


def get_glove_embeddings(path: str, index2word, save_path):
    """
    :param path: path to glove embedding text file
    :param index2word: index to word list
    :param save_path: path to save glove embedding pth file so U won't have to process again
    :return: tensor of size (vocab_size, embedding_dim)
    """
    with open('%s/glove.6B.300d.txt' % path, 'r') as file:
        text = file.read().split('\n')  # each starts with the character and following it its values

    # extract words and features from text file
    word_to_features = {}
    for line in text:
        line = line.split(' ')
        word_to_features[line[0]] = [float(num) for num in line[1:]]

    glove_embeddings = torch.zeros((len(index2word), 300), dtype=torch.float)
    for i, word in enumerate(tqdm(index2word[2:])):
        if word in word_to_features.keys():
            glove_embeddings[i] = torch.tensor([word_to_features[word]], dtype=torch.float)

    glove_embeddings[0] = torch.mean(glove_embeddings[2:], dim=0)
    torch.save(glove_embeddings, '%s.pth' % save_path)

    return glove_embeddings


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
        qu_ids, qu_list, im_list, im_box, ans_list = zip(*data)

        # pads questions to max question length in current batch
        self.qu = pad_sequence(qu_list, padding_value=pad_value).permute(1, 0)
        self.im = torch.stack(im_list, dim=0)
        self.im_box = torch.stack(im_box, dim=0)
        self.ans = torch.stack(ans_list, dim=0)
        self.qu_ids = qu_ids

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
