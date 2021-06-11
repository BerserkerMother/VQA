import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class New_Net(nn.Module):
    def __init__(self, num_classes, d_model, dropout, word_embedding, num_layers, num_heads):
        super(New_Net, self).__init__()

        """
        for question part we going to need a transformer encoder, thus transformer encoder layer
        for image and vision together we going to need transformer encoder again
        """
        self.hidden_size = d_model

        # word embedding
        num_emd, emd_dim = word_embedding.size()
        self.embedding = nn.Embedding(num_emd, emd_dim)
        self.embedding.load_state_dict(word_embedding)

        self.qu_token = nn.Parameter(torch.zeros(1, 1, d_model))

        question_modules = []
        for i in range(num_layers):
            question_modules.append(EncoderLayer(d_model, dropout, num_heads))
        self.question_encoder = nn.ModuleList(question_modules)

        # self attention layer
        module_list = []
        for i in range(num_layers):
            module_list.append(EncoderLayer(d_model, dropout, num_heads))
        self.self_attention_encoder = nn.ModuleList(module_list)

        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, questions, images_features, mask):
        batch_size, quN = questions.size()[:2]
        imN = images_features.size()[1]

        # reshape padding masks
        qu_mask, mixed_mask = self.generate_masks(mask, batch_size, quN, imN)

        x = self.embedding(questions)

        qu_token = self.qu_token.expand(batch_size, 1, self.hidden_size)
        x = torch.cat([qu_token, x], dim=1)

        for module in self.question_encoder:
            x = module(x, x, qu_mask)

        # cat image and question together
        x = torch.cat([x, images_features], dim=1)

        for module in self.self_attention_encoder:
            x = module(x, x, mixed_mask)

        output = self.decoder(x)

        return output

    def generate_masks(self, mask, batch_size, quN, imN):

        # generate question mask
        question_mask = mask.expand(batch_size, quN, quN)
        # generate mixed mask
        image_mask = torch.ones((batch_size, imN))
        mixed_mask = torch.cat([mask, image_mask], dim=1)
        mixed_mask = mixed_mask.expand(batch_size, quN + imN, quN + imN)

        return question_mask, mixed_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout, num_heads):
        """

        :param d_model:
        :param dropout:
        pipeline : multi head attention, add & norm, fc, add & norm
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y, mask):
        x1 = x
        x = self.multi_head_attention(x, y, y, mask)
        x = self.dropout(x) + x1
        x = self.norm1(x)

        x = self.mlp(x) + x
        x = self.norm2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        # input: tensor(B, num_obj, d_model)

        assert d_model % num_heads == 0
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.hidden_size = d_model

        # q k v linear projections
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size = q.size()[0]

        q = self.q(q).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.k(k).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = self.v(v).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_size)
        scores[mask] = float('-inf')

        attention_weights = F.softmax(scores, dim=3)

        attended_features = torch.matmul(attention_weights, v)
        attended_features = attended_features.permute(0, 2, 1, 3).contiguous.view(batch_size, -1, self.hidden_size)

        attended_features = self.linear(attended_features)

        return attended_features


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
