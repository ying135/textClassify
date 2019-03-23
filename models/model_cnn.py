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
        self.Ks = Ks
        Co = opt.hidden_size
        self.opt = opt
        if opt.use_embedvector:
            with open(os.path.join(opt.root,'embedmatrix.pkl'), 'rb') as f:
                embedmatri = pickle.load(f)
            embedmatrix = torch.Tensor(embedmatri)
            self.embed = nn.Embedding(opt.voca_length, embed_dim, _weight=embedmatrix)
        else:
            self.embed = nn.Embedding(opt.voca_length, embed_dim)
        # self.embed = nn.Embedding(, 300, _weight=opt.embedding_matrix)
        self.convs = nn.ModuleList([nn.Conv2d(1, Co, (K, embed_dim), padding=(K-1, 0)) for K in Ks])
        self.fc = nn.Sequential(nn.Linear(len(Ks) * Co, Co),
                                nn.ReLU(),
                                nn.Linear(Co, opt.num_class))
        if self.opt.attention is not None:
            self.attention = getattr(models, opt.attention)(Co)
            self.linear1 = nn.Linear(embed_dim, Co)
            self.linear2 = nn.Linear(len(Ks) * Co, Co)
            self.linear3 = nn.Linear(Co, opt.num_class)

    def forward(self, x):
        x = self.embed(x)
        embeds = x
        x = x.transpose(0, 1).unsqueeze(1)  # (batch, 1, seq_len, emded_dim)
        context = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # (batch, Co, seq_len) * len(Ks)
        # i'm not sure about the x and context
        if self.opt.attention is not None:
            outputs = []
            for emb in embeds.split(1):
                emb = self.linear1(emb)
                outp = [self.attention(emb, con.transpose(1, 2)) for con in context]
                outputs += outp
            outputs = torch.stack(outputs)
            outputs = outputs.contiguous().view(len(self.Ks), -1, outputs.size(1), outputs.size(2))
            output = [F.max_pool1d(i.permute(1, 2, 0), i.size(0)).squeeze(2) for i in outputs]
            output = torch.cat(output, 1)  # (batch, Co*len(Ks))
            output = self.fc(output)
            return output

        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in context]
        output = torch.cat(output, 1)     # (batch, Co*len(Ks))
        output = self.fc(output)
        return output


