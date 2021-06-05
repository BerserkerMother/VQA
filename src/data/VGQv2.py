# main dataset implementation modules

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

from collections import Counter


class VQA(Dataset):
    """
    dataset outputs image, question and answer index
    """

    def __init__(self, root: str):
        """

        :param root: path to root of which datasets are stored
        """
        self.root = root
