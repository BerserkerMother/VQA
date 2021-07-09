import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class New_Net(nn.Module):
    def __init__(self, num_classes, d_model, attention_dim, dropout, word_embedding, num_layers, num_heads):
        """

        :param num_classes: num of decoder output features
        :param d_model: model dimension
        :param dropout: dropout probability
        :param word_embedding: glove word embedding tensor
        :param num_layers: encoder number of layers, same for both encoders
        :param num_heads: number of heads in multi head attention
        """
        super(New_Net, self).__init__()
        self.d_model = d_model
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        self.embedding = Embedding(d_model, word_embedding, dropout)  # qu embedding
        self.image_embedding = ImageEmbedding(d_model, 2048, 4, dropout)  # im embedding

        # cls token
        self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)))

        question_modules = []
        for i in range(num_layers):
            question_modules.append(EncoderLayer(d_model, attention_dim, dropout, num_heads))
        self.question_encoder = nn.ModuleList(question_modules)

        image_modules = []
        for i in range(num_layers):
            image_modules.append(EncoderLayer(d_model, attention_dim, dropout, num_heads))
        self.image_encoder = nn.ModuleList(image_modules)

        # self attention layer
        module_list = []
        for i in range(num_layers):
            module_list.append(EncoderLayer(d_model, attention_dim, dropout, num_heads))
        self.self_attention_encoder = nn.ModuleList(module_list)

        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, questions, images_features, image_box, mask):
        """

        :param questions: question batch of size (batch_size, question_length, embedding dim)
        :param images_features: image features of size(batch_size, num_objects=36, 2048)
        :param image_box: location of image objects
        :param mask: question padding mask, same size as question
        :return:
        """
        batch_size = questions.size()[0]

        x = self.embedding(questions)
        y = self.image_embedding(images_features, image_box)

        # add new token to padding mask
        mask = torch.cat([torch.zeros((batch_size, 1), device=torch.device('cuda:0')), mask], dim=1)
        quN = x.size()[1]
        imN = y.size()[1]

        # reshape padding masks
        qu_mask, mixed_mask = self.generate_masks(mask, batch_size, quN, imN)

        for module in self.question_encoder:
            x = module(x, x, qu_mask)

        for module in self.image_encoder:
            y = module(y, y)

        # cat image and question together & add cls token
        cls_token = self.cls_token.expand(batch_size, 1, self.d_model)
        x = torch.cat([cls_token, x, y], dim=1)

        for module in self.self_attention_encoder:
            x = module(x, x, mixed_mask)

        output = self.decoder(x[:, 0])

        return output

    def generate_masks(self, mask, batch_size, quN, imN):
        """

        :param mask: question mask
        :param batch_size:
        :param quN: question length
        :param imN: num image objects
        :return: given input, outputs resized and expanded mask for question and qu&im encoder
        """

        # TODO: check to see if masks are correct
        mask = mask.view(batch_size, 1, quN, 1)
        # generate question mask
        question_mask = mask.expand(batch_size, self.num_heads, quN, quN)
        question_mask = question_mask.permute(0, 1, 3, 2)
        # generate mixed mask
        cls_token_mask = torch.zeros((batch_size, 1, 1, 1), device=torch.device('cuda:0'))
        image_mask = torch.zeros((batch_size, 1, imN, 1), device=torch.device('cuda:0'))
        mixed_mask = torch.cat([cls_token_mask, mask, image_mask], dim=2)
        mixed_mask = mixed_mask.expand(batch_size, self.num_heads, quN + imN + 1, quN + imN + 1)
        mixed_mask = mixed_mask.permute(0, 1, 3, 2)

        return question_mask == True, mixed_mask == True


class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention_dim, dropout, num_heads):
        """

        :param d_model:
        :param dropout:
        pipeline : multi head attention, add & norm, fc, add & norm
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, attention_dim, num_heads, dropout)
        self.mlp = MLP(d_model, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y, mask=None):
        x1 = x
        x = self.multi_head_attention(x, y, y, mask)
        x = self.dropout(x) + x1
        x = self.norm1(x)

        x = self.mlp(x) + x
        x = self.norm2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, attention_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        # input: tensor(B, num_obj, d_model)

        assert attention_dim % num_heads == 0
        self.head_size = attention_dim // num_heads
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.d_model = d_model

        # q k v linear projections
        self.q = nn.Linear(d_model, attention_dim)
        self.k = nn.Linear(d_model, attention_dim)
        self.v = nn.Linear(d_model, attention_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(attention_dim, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size()[0]

        q = self.q(q).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.k(k).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = self.v(v).view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(mask, (float('-inf')))

        attention_weights = F.softmax(scores, dim=3)
        attention_weights = self.dropout(attention_weights)

        attended_features = torch.matmul(attention_weights, v)
        attended_features = attended_features.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.attention_dim)

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


class ImageEmbedding(nn.Module):
    def __init__(self, d_model, feature_dim, pos_dim, dropout=.2):
        super(ImageEmbedding, self).__init__()
        self.d_model = d_model

        self.im_linear = nn.Linear(feature_dim, d_model)
        self.pos_linear = nn.Linear(pos_dim, d_model)

        self.im_norm = nn.LayerNorm(d_model)
        self.p_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.im_token = nn.Parameter(torch.zeros((1, 1, d_model)))

    def forward(self, x, pos_x):
        batch_size = x.size()[0]
        x = self.im_norm(self.im_linear(x)) + self.p_norm(self.pos_linear(pos_x))
        x = self.dropout(x * .5)

        im_token = self.im_token.expand(batch_size, 1, self.d_model)
        x = torch.cat((im_token, x), dim=1)

        return x


class Embedding(nn.Module):
    def __init__(self, d_model, word_embedding, dropout=.2):
        super(Embedding, self).__init__()

        self.d_model = d_model
        # class token
        self.qu_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # word embedding
        num_emd, emd_dim = word_embedding.size()
        self.word_embedding = nn.Embedding(num_emd, emd_dim, _weight=word_embedding)
        self.qu_fc = nn.Linear(emd_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model)

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.word_embedding(x)
        x = self.qu_fc(x)
        x = self.pos_embedding(x)
        x = self.norm(x)
        x = self.dropout(x)

        qu_token = self.qu_token.expand(batch_size, 1, self.d_model)
        x = torch.cat([qu_token, x], dim=1)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.size()[1]]
        return x
