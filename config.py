import torch


class DefaultConfig(object):
    train = True
    load_model_path = None
    data_ratio = 1 # the proportion of trainset

    epoch = 100
    batch_size = 64
    num_worker = 4
    lr = 0.001
    weight_decay = 0e-5
    lr_decay = 0.5
    print_inter = 50
    hidden_size = 128

    num_class = 5

    embed_dim = 300

    model_name = 'CLSTM'
    attention = 'global_attention'

    voca_length = 33748
    use_embedvector = True

    use_cuda = torch.cuda.is_available()
    root = 'data'

    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
