# main dataset implementation modules

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

import os
import json
from glob import glob

from .utils import *


class VQA(Dataset):
    """
    dataset contains a dictionary {question id, question}, {image id, image_path}, {question id image id: answer}
    """

    def __init__(self, root: str, min_ans_freq: int = 10):
        """

        :param root: path to root of which datasets are stored
        :param min_ans_freq: minimum answer frequency to include sample
        """
        self.root = root
        self.min_ans_freq = min_ans_freq

        self.tokenizer = get_tokenizer('basic_english')

        # default paths
        annotations_path = ['annotations/v2_mscoco_train2014_annotations.json',
                            'annotations/v2_mscoco_val2014_annotations.json']
        questions_path = ['questions/v2_OpenEnded_mscoco_train2014_questions.json',
                          'questions/v2_OpenEnded_mscoco_val2014_questions.json']
        images_path = ['mscoco_imgfeat/train2014', 'mscoco_imgfeat/val2014']

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
        self.vocab_stoi = self.vocab.get_stoi()
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

        print('creating answer bank...')
        # dictionary mapping each answer to its index
        self.answer2index = self.answer_bank()

        print('creating data triplets')
        self.data = self.make_triplets()

        print('data processing is Done\ndataset contains %d samples and %d unique answers' % (
            len(self.data), len(self.answer2index)))

    def __getitem__(self, idx):
        qu_id, im_id, ans_idx = self.data[idx]

        question = self.questions[qu_id]
        image = torch.tensor(np.load(self.images[im_id]), dtype=torch.float)

        return question, image, int(ans_idx)

    def __len__(self):
        return len(self.data)

    # convert str question to tensors, each word is represented by its index in vocab
    def questions_as_tensor(self):
        question_ten = {}
        for qu_id in self.questions:
            qu = self.questions[qu_id]
            temp = []

            for word in self.tokenizer(qu):
                if word in self.vocab_stoi:
                    temp.append(self.vocab_stoi[word])
                else:
                    temp.append(self.vocab_stoi['<unk>'])
            question_ten[qu_id] = torch.tensor(temp, dtype=torch.long)

        return question_ten

    # extracts all the answers from annotations and makes the classes for network
    def answer_bank(self):
        counter = Counter()
        count = 0
        answer2index = {}

        for item in self.annotations:
            answer = item['multiple_choice_answer']
            counter.update([answer])

        for answer in counter.keys():
            if counter[answer] >= self.min_ans_freq:
                answer2index[answer] = count
                count += 1

        return answer2index

    # makes triplets of data (qu_id, im_id, ans_idx)
    def make_triplets(self):
        data = []
        for item in self.annotations:
            answer = item['multiple_choice_answer']

            if answer in self.answer2index.keys():
                answer_id = self.answer2index[answer]
                question_id = item['question_id']
                image_id = item['image_id']

                data.append((str(question_id), str(image_id), answer_id))

        return data
