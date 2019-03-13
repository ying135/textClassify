import torch
from torch.utils import data
import linecache


# many to one, the first edition of this class is based on en_vi translation
# exactly the same with many 2 many
class m2odata(data.Dataset):
    def __init__(self, roots):
        self.srcindf = roots['srcindf']
        self.tgtindf = roots['tgtindf']
        self.srcstrf = roots['srcstrf']
        self.tgtstrf = roots['tgtstrf']
        self.length = roots['length']


    def __getitem__(self, item):
        item = item + 1
        srcind = list(map(int, linecache.getline(self.srcindf, item).strip().split()))
        tgtind = list(map(int, linecache.getline(self.tgtindf, item).strip().split()))
        # didn't consider that whether char level
        srcstr = linecache.getline(self.srcstrf, item).strip().split()
        tgtstr = linecache.getline(self.tgtstrf, item).strip().split()

        return srcind, tgtind, srcstr, tgtstr

    def __len__(self):
        return self.length

# many2one padding
def padding(data):
    srcind, tgtind, srcstr, tgtstr = zip(*data)

    src_len = [len(s) for s in srcind]
    src_pad = torch.zeros(len(srcind), max(src_len)).long()
    for i, s in enumerate(srcind):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[:end])

    tgt_pad = torch.LongTensor(tgtind).squeeze(1)

    return src_pad, tgt_pad, torch.Tensor(src_len).long(), None, srcstr, tgtstr

