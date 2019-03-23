import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import models


class textCnn(torch.nn.Module):
    def __init__(self, opt):
        super(textCnn, self).__init__()

        embed_dim = opt.embed_dim
        Ks = (3, 4, 5)
        Co = 100
        if opt.use_embedvector:
            with open(os.path.join(opt.root,'embedmatrix.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length, embed_dim)
        # self.embed = nn.Embedding(, 300, _weight=opt.embedding_matrix)
        self.convs = nn.ModuleList([nn.Conv2d(1, Co, (K, embed_dim)) for K in Ks])
        self.fc = nn.Sequential(nn.Linear(len(Ks) * Co, Co),
                                nn.ReLU(),
                                nn.Linear(Co, opt.num_class))

    def forward(self, x):
        x = x.t()   # (batch, seq_len)
        x = self.embed(x)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, emded_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # (batch, Co, seq_len) * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)     # (batch, Co*len(Ks))
        output = self.fc(x)
        return output


