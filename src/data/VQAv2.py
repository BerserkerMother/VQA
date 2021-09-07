# main dataset implementation modules

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchvision import transforms

import os
from glob import glob
from PIL import Image


from .utils import *

T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class VQA(Dataset):
    """
    dataset contains a dictionary {question id, question}, {image id, image_path}, {question id image id: answer}
    """

    def __init__(self, root: str, split: str, ans_json: str, min_ans_freq: int = 10,
                 max_qu_length: int = 14, answer_bank=None, vocab=None, resnet: bool = False):
        """

        :param root: path to root of which datasets are stored
        :param min_ans_freq: minimum answer frequency to include sample
        """
        self.root = root
        self.min_ans_freq = min_ans_freq
        self.max_qu_length = max_qu_length
        self.ans_json = ans_json
        self.split = split
        self.resnet = resnet

        self.tokenizer = get_tokenizer('basic_english')

        # default paths
        annotations_path = ['annotations/v2_mscoco_train2014_annotations.json',
                            'annotations/v2_mscoco_val2014_annotations.json']
        questions_path = ['questions/v2_OpenEnded_mscoco_train2014_questions.json',
                          'questions/v2_OpenEnded_mscoco_val2014_questions.json']
        images_path = ['mscoco_imgfeat/train2014', 'mscoco_imgfeat/val2014']

        if self.split == 'train':
            annotations_path = [annotations_path[0]]
            questions_path = [questions_path[0]]
            images_path = [images_path[0]]

        if self.split == 'val':
            annotations_path = [annotations_path[1]]
            questions_path = [questions_path[1]]
            images_path = [images_path[1]]

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
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = make_vocab(self.questions, self.tokenizer)
        self.vocab_stoi = self.vocab.get_stoi()
        self.questions = self.questions_as_tensor()

        self.images = {}
        if resnet:
            if self.split == 'train':
                path = '~/VQA/data/mscoco/train2014'
            else:
                path = '~/VQA/data/mscoco/val2014'

            for image_path in glob(path + '/*'):
                image_id = image_path.split('/')[-1].split('_')[-1].split('.')[0]
                self.images[str(int(image_id))] = image_path
            print('%s loaded' % self.split)
        else:
            for path in images_path:
                path = os.path.join(self.root, path)
                if os.path.exists(path):
                    for image_path in glob(path + '/*'):
                        image_id = image_path.split('/')[-1].split('_')[-1].split('.')[0]
                        if image_id[-1] != 'l':
                            self.images[str(int(image_id))] = image_path
                    print('%s loaded to images' % path, end=' | ')
        print('images loaded')

        print('creating answer bank...')
        # dictionary mapping each answer to its index
        if answer_bank:
            self.answer2index, self.index2_answer = answer_bank
        else:
            self.answer2index, self.index2_answer = self.answer_bank()

        print('creating data triplets')
        self.data = self.make_triplets()

        print('data processing is Done\ndataset contains %d samples and %d unique answers' % (
            len(self.data), len(self.answer2index)))

    def __getitem__(self, idx):
        qu_id, im_id, ans_idx = self.data[idx]

        question = self.questions[qu_id]
        if self.resnet:
            image = Image.open(self.images[im_id]).convert(mode='RGB')
            image = T(image)
            image_box = torch.tensor([1])
        else:
            image = torch.tensor(np.load(self.images[im_id]), dtype=torch.float)
            image_box = torch.tensor(np.load('%sl.npy' % (self.images[im_id][:-4])), dtype=torch.float)

        return question, image, image_box, int(ans_idx), qu_id

    def __len__(self):
        return len(self.data)

    def questions_as_tensor(self):
        """

        :return: convert str question to tensors, each word is represented by its index in vocab
        """
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

    def answer_bank(self):
        """

        :return: extracts all the answers from annotations and makes the classes for network
        """
        if self.ans_json:
            return get_answer_dict(self.ans_json)

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
        index2answer = [value for value in answer2index.values()]

        return answer2index, index2answer

    def make_triplets(self):
        """

        :return: makes triplets of data (qu_id, im_id, ans_idx)
        """
        data = []
        for item in self.annotations:
            answer = item['multiple_choice_answer']

            if answer in self.answer2index.keys():
                answer_id = self.answer2index[answer]
                question_id = str(item['question_id'])
                image_id = str(item['image_id'])

                if len(self.questions[question_id]) <= self.max_qu_length:
                    data.append((question_id, image_id, answer_id))

        return data
