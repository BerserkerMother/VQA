# data utility modules
import torch
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from tqdm import tqdm


# takes questions vocab and GloVe text and returns tensor containing weights
def get_glove_embeddings(path, question_vocab):
    with open(path, 'r') as file:
        text = file.read().split('\n')  # each starts with the character and following it its values

    # extract words and features from text file
    word_to_features = {}
    for line in text:
        line = line.split(' ')
        word_to_features[line[0]] = [float(num) for num in line[1:]]

    # make desired glove embedding
    glove_embeddings = torch.tensor([])
    for word in tqdm(question_vocab.itos):
        if word in word_to_features.keys():
            temp = torch.tensor([word_to_features[word]])
        else:
            temp = torch.zeros((1, 50))

        glove_embeddings = torch.cat((glove_embeddings, temp))

    return glove_embeddings


# given questions makes vocab
def make_vocab(questions, tokenizer):
    counter = Counter()
    for question in questions.values():
        counter.update(tokenizer(question))
    vocab = Vocab(counter, specials=['<pad>', '<unk>'])

    return vocab


def get_collate_fn(pad_value):
    def generate_padding(batch):
        qu, im, ans = batch
        qu = pad_sequence(qu, padding_value=pad_value)

        return qu, im, ans, qu==True

    return generate_padding
