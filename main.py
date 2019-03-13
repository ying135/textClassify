import torch
from torch.utils.data import DataLoader
import config
import pickle
import os
import datasetmaker
import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

opt = config.DefaultConfig()

# many2one version
def load_data(train):
    print("loading data...")
    with open(os.path.join(opt.root, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    dict_src = data['dicts']['src']

    if train:
        data['train']['length'] = int(data['train']['length'] * opt.data_ratio)
        data['valid']['length'] = int(data['valid']['length'] * opt.data_ratio)
        trainset = datasetmaker.m2odata(data['train'])
        validset = datasetmaker.m2odata(data['valid'])
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=opt.num_worker, collate_fn=datasetmaker.padding)
        validloader = DataLoader(validset, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=opt.num_worker, collate_fn=datasetmaker.padding)
        # the loader generate the batch which contains: src_pad, tgt_pad, src_len, tgt_len, srcstr, tgtstr
        # src_pad and tgt_pad dimension : (batch_size, max(src_len) in this batch)
        # src_len: length of sentence(after being filtered and trunc)  dimension: (batchsize)
        # tgt_len: length of sentence(after being filtered and trunc) plus 2(bos eos) dimension: (batchsize)
        # srcstr, tgtstr : the string with no padding
        return trainloader, validloader, dict_src
    else:
        data['test']['length'] = int(data['test']['length'] * opt.data_ratio)
        testset = datasetmaker.m2mdata(data['test'])
        testloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)
        return testloader, dict_src

def plotlc(x, y, figname='learning_curve'):
    plt.plot(x, y)
    plt.title('learning curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig(figname)

def train(model, trainloader, validloader):
    if opt.use_cuda:
        model = model.cuda()
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=opt.weight_decay)
    previous_loss = 1e10
    pEpoch = []
    pLoss = []

    for epoch in range(opt.epoch):
        loss_all = 0
        total_accuracy = 0

        for i, (input, target, src_len, tgt_len, inputstr, targetstr) in enumerate(trainloader):
            if opt.use_cuda:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            pred = torch.max(score, 1)[1]
            accuracy = float((pred == target).sum())
            accuracy = accuracy * 100 / input.size(0)

            total_accuracy += accuracy
            loss_all += float(loss)

            if i % opt.print_inter ==0:
                print("Epoch: ", epoch, "| Iter:", i, "| Loss:", float(loss), "| Accuracy:", accuracy, "%")

        avgloss = loss_all / len(trainloader)
        avgaccuracy = total_accuracy / len(trainloader)
        print("the end of Epoch: ", epoch, "| AVGLoss:", avgloss, "| Accuracy:", avgaccuracy, "%")
        pEpoch.append(epoch)
        pLoss.append(avgloss)
        plotlc(pEpoch, pLoss)

        # update lr
        if avgloss > previous_loss:
            lr = lr * opt.lr_decay
            for para in optimizer.param_groups:
                para['lr'] = lr
            print("learning rate changes to ",lr)
        previous_loss = avgloss



def valid(model, validloader):
    pass


def test(model, testloader):
    pass


def main():
    model = models.textCnn(opt)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path))
        print("Load Success!", opt.load_model_path)

    if opt.train:
        trainloader, validloader, dict_src = load_data(opt.train)
        opt.parse({'voca_length': len(dict_src)})
        print("START training...")
        train(model, trainloader, validloader)
    else:
        testloader, dict_src = load_data(opt.train)
        opt.parse({'voca_length': len(dict_src)})
        print("START testing...")
        test(model, testloader)


if __name__ == "__main__":
    main()
