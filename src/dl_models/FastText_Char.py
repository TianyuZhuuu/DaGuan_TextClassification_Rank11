import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from src.dl_models.utils import cross_validation_bagging, hold_out_test


class FastText(nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super(FastText, self).__init__()
        embed_mat = torch.from_numpy(np.load(f'../../data/char_embed_mat.npy'))
        num_word, embed_dim = embed_mat.size()
        self.embed = nn.Embedding.from_pretrained(embed_mat)
        self.fc1 = nn.Linear(embed_dim, fc_dim1, bias=False)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2, bias=False)
        self.out = nn.Linear(fc_dim2, 19)
        self.act = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(fc_dim1)
        self.bn2 = nn.BatchNorm1d(fc_dim2)

    def forward(self, input):
        out = self.embed(input)

        out = torch.mean(out, dim=1)
        out = self.bn1(F.dropout(self.act(self.fc1(out)), p=0.5, training=self.training, inplace=True))
        out = self.bn2(F.dropout(self.act(self.fc2(out)), p=0.2, training=self.training, inplace=True))
        out = self.out(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # seed = 1
    model_fn = lambda: FastText(1024, 1024)
    model_name = 'fasttext_char'
    train_data = np.load('../../data/char_train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/char_test_input.npy')
    # hold_out_test(model_fn, model_name, train_data, train_label, test_data, lr=1e-4)
    cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, lr=1e-4, patience=20, seed=2)

    # seed = 1
    # model_fn = lambda: Fast_Attention_Text(1024, 1024)
    # model_name = 'fast_attention_text'
    # train_data = np.load('../../data/train_input.npy')
    # train_label = np.load('../../data/label.npy')
    # test_data = np.load('../../data/test_input.npy')
    # hold_out_test(model_fn, model_name, train_data, train_label, test_data)
    # cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, batch_size=32, lr=5e-4,
    #                          seed=seed)
