import torch
import torch.nn as nn
import torch.nn.functional as F
class textCnn(torch.nn.Module):
    def __init__(self, opt):
        super(textCnn, self).__init__()

        embed_dim = 100
        Ks = (3, 4, 5)
        Co = 100

        self.embed = nn.Embedding(opt.voca_length, embed_dim)
        # self.embed = nn.Embedding(, 300, _weight=opt.embedding_matrix)
        self.convs = nn.ModuleList([nn.Conv2d(1, Co, (K, embed_dim)) for K in Ks])
        self.fc = nn.Sequential(nn.Linear(len(Ks) * Co, Co),
                                nn.ReLU(),
                                nn.Linear(Co, 5))


    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)     # (batch, Co*len(Ks))
        output = self.fc(x)
        return output


