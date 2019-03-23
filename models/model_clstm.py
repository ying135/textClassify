import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import models
import pickle
import os


class CLSTM(nn.Module):
    def __init__(self, opt):
        super(CLSTM, self).__init__()

        self.opt = opt
        C = opt.num_class
        embed_dim = opt.embed_dim
        hidden_size = opt.hidden_size
        kernel_size = 3

        if opt.use_embedvector:
            with open(os.path.join(opt.root, 'embedmatrix.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length, embed_dim)

        self.conv1 = nn.Conv2d(1, hidden_size, (kernel_size, embed_dim))
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 2, C))
        if self.opt.attention is not None:
            self.attention = getattr(models, opt.attention)(hidden_size)
            self.linear1 = nn.Linear(embed_dim, hidden_size)

    def forward(self, x):
        x = self.embed(x)   # (seq_len, batch, embed_dim)
        embeds = x
        x = x.transpose(0, 1).unsqueeze(1)
        x = self.conv1(x).squeeze(3)
        x = x.transpose_(0, 2).transpose_(1, 2) # (seq_Len, batch, hidden_size)
        r_out, h_state = self.rnn(x)
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




