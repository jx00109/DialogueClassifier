# -*- encoding:utf8 -*-
import torch
import torch.autograd as autograd  # torch中自动计算梯度模块
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络模块中的常用功能
import torch.optim as optim  # 模型优化器模块

torch.manual_seed(1)


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.init_emb()
        # self.hidden = self.init_hidden()

    # 在0附件初始化词向量矩阵
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    '''
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    '''

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # x: (time_step, batch, embedding_dim)
        lstm_out, hidden = self.lstm(embeds.view(len(sentence), 1, -1), None)

        tag_space = self.hidden2tag(lstm_out[-1, :, :])

        # tag_scores = F.softmax(tag_space)
        return tag_space
