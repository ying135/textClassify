import os

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import models
import pickle
import os


class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        C = opt.num_class
        embed_dim = opt.embed_dim
        hidden_size = opt.hidden_size

        if opt.use_embedvector:
            with open(os.path.join(opt.root, 'embedmatrix.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length, embed_dim)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                nn.ReLU(),
                                nn.Linear(hidden_size//2, C))

    def forward(self, x):
        x = self.embed(x)   # (seq_len, batch, embed_dim)
        r_out, h_state = self.rnn(x)
        # (seq_len, batch, hidden_size) h/c (1, batch, hidden_size)
        output = self.fc(h_state[0].squeeze(0))
        return output
