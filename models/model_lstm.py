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
        if self.opt.attention is not None:
            self.attention = getattr(models, opt.attention)(hidden_size)
            self.linear1 = nn.Linear(embed_dim, hidden_size)

    def forward(self, x):
        x = self.embed(x)   # (seq_len, batch, embed_dim)
        embeds = x
        r_out, h_state = self.rnn(x)
        # (seq_len, batch, hidden_size) h/c (1, batch, hidden_size)
        if self.opt.attention is not None:
            outputs = []
            for emb in embeds.split(1):
                emb = self.linear1(emb)
                outp = self.attention(emb, r_out.transpose(0, 1))
                outputs += [outp]
            outputs = torch.stack(outputs)  # (seq_len, batch, hidden_size)
            output = outputs.permute(1, 2, 0)
            output = F.max_pool1d(output, output.size(2)).squeeze(2)
            output = self.fc(output)
            return output

        output = self.fc(h_state[0].squeeze(0))
        return output
