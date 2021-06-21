# data utility modules
import torch
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from tqdm import tqdm


# takes questions vocab and GloVe text and returns tensor containing weights
def get_glove_embeddings(path, question_vocab, save_path):
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
    counter = Counter()
    for question in questions.values():
        counter.update(tokenizer(question))
    counter.update(['<pad>', '<unk>'])
    v = vocab(counter, min_freq=1)

    return v


def get_collate_fn(pad_value):
    def generate_padding(batch):
        qu_list, im_list, ans_list = [], [], []
        for qu, im, ans in batch:
            qu_list.append(qu)
            im_list.append(im)
            ans_list.append(ans)

        qu = pad_sequence(qu_list, padding_value=pad_value).permute(1, 0)

        return qu, torch.stack(im_list, dim=0), torch.tensor(ans_list, dtype=torch.long), (qu == pad_value)

    return generate_padding
