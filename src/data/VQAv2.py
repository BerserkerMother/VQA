# VQA v2 dataset implementation using extracted bottom up features
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

import glob
import os
import json
import numpy as np

from .utils import get_vocab, get_candidate_answers, is_answers_in_bank, get_ans_scores, get_answers_array
from .path import VQAv2_FILENAMES


class VQADataset(Dataset):
    def __init__(self, root: str, splits: str = 'train+val', vocab: Vocab = None, candidate_answers: tuple = None,
                 max_questions_length: int = 14):
        """

        :param root: path to data dir
        :param splits: define the training sets
        :param vocab: Vocab object
        :param candidate_answers: tuple of (answer2index, index2answer)
        :param max_questions_length: maximum length of question that is allowed
        """

        self.root = root
        self.splits = splits.split('+')

        print('processing %s set\nreading json files...' % splits)
        self.annotations, self.questions = {'annotations': []}, {'questions': []}
        self.im_id2im_feat_path, self.im_id2im_box_path, self.qu_id2qu_text = {}, {}, {}
        # add all annotations and question JSON files to lists
        # creates (im_id, image_feat_path) & (im_id, image_box_path) dictionary
        for split in self.splits:
            # read corresponding annotations
            annotation_path = os.path.join(self.root, VQAv2_FILENAMES[split + '_an'])
            annotations = json.load(open(annotation_path, 'r'))['annotations']
            self.annotations['annotations'] += annotations
            # red corresponding questions
            question_path = os.path.join(self.root, VQAv2_FILENAMES[split + '_qu'])
            questions = json.load(open(question_path, 'r'))['questions']
            self.questions['questions'] += questions
            for question in questions:
                self.qu_id2qu_text[str(question['question_id'])] = question['question']

            # create image dic
            im_dir_path = os.path.join(self.root, VQAv2_FILENAMES[split + '_im'])
            for path in glob.glob(im_dir_path + 'features/*'):
                im_id = str(int(path.split('/')[-1].split('.')[0].split('_')[-1]))
                self.im_id2im_feat_path[im_id] = path

            for path in glob.glob(im_dir_path + 'box/*'):
                im_id = str(int(path.split('/')[-1].split('.')[0].split('_')[-1]))
                self.im_id2im_box_path[im_id] = path
        # check if number of image features  and image box files match
        assert (len(self.im_id2im_feat_path) == len(self.im_id2im_box_path)), \
            'number of image features and boxed doesn\'t match, please check the image files'
        print('DONE!')

        # create tokenizer
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        print('building the vocabulary...')
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = get_vocab(self.questions, tokenizer)

        self.word2index = self.vocab.get_stoi()
        self.index2word = self.vocab.get_itos()
        print('DONE!\nvocab size: %d' % len(self.index2word))

        # use candidate answer if it's provided
        print('preparing candidate answers...')
        if candidate_answers:
            self.answer2index, self.index2answer = candidate_answers
        else:
            self.answer2index, self.index2answer = get_candidate_answers(self.annotations)
        print('DONE!\nnumber of answers: %d' % len(self.index2answer))

        print('making data file list...')
        # creating list of data tuples(qu_id, im_id)
        self.data = []
        # creating (qu_id: question_tensor),(qu_i:, ans_scores) dictionaries
        self.qu_id2qu_tensor, self.qu_id2ans_scr = {}, {}
        # modifying (im_id: image_feat_path) & (im_id: image_box_path)

        # make new question and annotation json for evaluation
        annotations, questions = {'annotations': []}, {'questions': []}
        for i, annotation in enumerate(self.annotations['annotations']):
            question_id = str(annotation['question_id'])
            image_id = str(annotation['image_id'])
            answers = annotation['answers']

            question_text = tokenizer(self.qu_id2qu_text[question_id])

            # checks if one of the answers is in the candidate answers and question length doesn't surpass the limit
            ans = get_answers_array(answers)
            if len(question_text) <= max_questions_length and is_answers_in_bank(ans, self.answer2index):
                self.data.append((question_id, image_id))
                token_idx = []
                for token in question_text:
                    token_idx.append(self.word2index[token] if token in self.word2index else self.word2index['<unk>'])

                self.qu_id2qu_tensor[question_id] = torch.tensor(token_idx, dtype=torch.long)
                self.qu_id2ans_scr[question_id] = ans

                annotations['annotations'].append(annotation)

        for question in self.questions['questions']:
            if str(question['question_id']) in self.qu_id2qu_text:
                questions['questions'].append(question)
        print('DONE!')

        print('saving new annotations and questions to %s' % self.root)
        # save new questions and annotations
        json.dump(questions, open(self.root + '%s_questions.json' % splits, 'w'))
        json.dump(annotations, open(self.root + '%s_annotations.json' % splits, 'w'))
        print('DONE!\nall set, let\'s do some deep learning')
        print('_' * 20)

    def __len__(self):
        return len(self.data)

    # make answer scores online
    def __getitem__(self, idx):
        qu_id, im_id = self.data[idx]

        qu_tensor = self.qu_id2qu_tensor[qu_id]
        im_feat = torch.tensor(np.load(self.im_id2im_feat_path[im_id]), dtype=torch.float)
        im_box = torch.tensor(np.load(self.im_id2im_box_path[im_id]), dtype=torch.float)
        answer_scores = torch.tensor(get_ans_scores(self.qu_id2ans_scr[qu_id], self.answer2index))

        return qu_id, qu_tensor, im_feat, im_box, answer_scores
