# data utility modules
import torch

from tqdm import tqdm


# takes questions vocab and GloVe text and returns tensor containing weights
def get_glove_embeddings(path, question_vocab):
    with open(path, 'r') as file:
        text = file.read().split('\n')  # each starts with the character and following it its values

    # extract words and features from text file
    word_to_features = {}
    glove_embeddings = torch.tensor([])
    for line in text:
        line = line.split(' ')
        word_to_features[line[0]] = [float(num) for num in line[1:]]

    for word in tqdm(question_vocab.itos):
        if word in word_to_features.keys():
            temp = torch.tensor([word_to_features[word]])
        else:
            temp = torch.zeros((1, 50))

        glove_embeddings = torch.cat((glove_embeddings, temp))

    return glove_embeddings
