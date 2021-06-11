# main dataset implementation modules

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

import os
import json
from PIL import Image
from glob import glob

from .utils import *


class VQA(Dataset):
    """
    dataset contains a dictionary {question id, question}, {image id, image_path}, {question id image id: answer}
    """

    def __init__(self, root: str):
        """

        :param root: path to root of which datasets are stored
        """
        self.root = root

        self.vocab = Vocab(Counter())
        self.tokenizer = get_tokenizer('basic_english')

        # default paths
        annotations_path = ['v2_mscoco_train2014_annotations.json', 'val']
        questions_path = ['v2_OpenEnded_mscoco_train2014_questions.json', 'val']
        images_path = ['train2014', 'val']

        self.annotations = []
        for path in annotations_path:
            path = os.path.join(self.root, path)
            if os.path.exists(path):
                self.annotations += json.load(open(path, 'r'))['annotations']
                print('%s loaded to annotations' % path, end=' | ')

        print('annotations loaded')

        self.questions = {}
        for path in questions_path:
            path = os.path.join(self.root, path)
            if os.path.exists(path):
                questions = json.load(open(path, 'r'))['questions']
                for question in questions:
                    qu_id = str(question['question_id'])
                    qu = question['question']
                    self.questions[qu_id] = qu

                print('%s loaded to questions' % path, end=' | ')
        print('questions loaded')
        self.vocab = make_vocab(self.questions, self.tokenizer)
        self.questions = self.questions_as_tensor()

        self.images = {}
        for path in images_path:
            path = os.path.join(self.root, path)
            if os.path.exists(path):
                for image_path in glob(path + '/*'):
                    image_id = int(image_path.split('/')[-1].split('_')[-1].split('.')[0])
                    self.images[str(image_id)] = image_path
                print('%s loaded to images' % path, end=' | ')

        print('images loaded')

        # dictionary mapping each answer to its index
        self.answer2index = self.answer_bank()

        self.data = self.make_triplets()

    def __getitem__(self, idx):
        qu_id, im_id, ans_idx = self.data[idx]

        question = self.questions[qu_id]
        image = Image.open(self.images[im_id])

        return question, image, ans_idx

    def __len__(self):
        return len(self.data)

    # convert str question to their tokenized indexes as tensor
    def questions_as_tensor(self):
        temp = {}
        for qu_id in self.questions:
            qu = self.questions[qu_id]
            temp[qu_id] = torch.tensor([self.vocab.stoi[word] for word in self.tokenizer(qu)], dtype=torch.long)

        return temp

    def answer_bank(self):
        count = 0
        answer2index = {}

        for item in self.annotations:
            answer = item['multiple_choice_answer']
            if answer not in answer2index.keys():
                answer2index[answer] = str(count)
                count += 1

        return answer2index

    def make_triplets(self):
        data = []
        for item in self.annotations:
            answer = item['multiple_choice_answer']

            answer_id = self.answer2index[answer]
            question_id = item['question_id']
            image_id = item['image_id']

            data.append((str(question_id), str(image_id), answer_id))

        return data
