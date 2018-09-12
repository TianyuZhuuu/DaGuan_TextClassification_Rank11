import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from src.dl_models.utils import cross_validation_bagging, hold_out_test


class FastText(nn.Module):
    def __init__(self, fc_dim1, fc_dim2, granularity='word'):
        super(FastText, self).__init__()
        embed_mat = torch.from_numpy(np.load(f'../../data/{granularity}_embed_mat.npy').astype(np.float32))
        num_word, embed_dim = embed_mat.size()
        self.embed = nn.Embedding.from_pretrained(embed_mat, False)
        self.fc1 = nn.Linear(embed_dim, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.out = nn.Linear(fc_dim2, 19)
        self.act = nn.RReLU()
        self.bn1 = nn.BatchNorm1d(fc_dim1)
        self.bn2 = nn.BatchNorm1d(fc_dim2)

    def forward(self, input):
        out = self.embed(input)
        out = torch.mean(out, dim=1)
        out = self.bn1(F.dropout(self.act(self.fc1(out)), p=0.5, training=self.training, inplace=True))
        out = self.bn2(F.dropout(self.act(self.fc2(out)), p=0.5, training=self.training, inplace=True))
        out = self.out(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Fast_Attention_Text(nn.Module):
    def __init__(self, fc_dim1, fc_dim2):
        super(Fast_Attention_Text, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        embed_mat = torch.from_numpy(np.load(f'../../data/word_embed_mat.npy').astype(np.float32))
        num_word, embed_dim = embed_mat.size()
        self.embed = nn.Embedding.from_pretrained(embed_mat, False)
        self.ctx_vec_size = (300,)
        self.ctx_vec = nn.Parameter(torch.randn(self.ctx_vec_size).float().to(self.device))
        self.proj = nn.Linear(in_features=300, out_features=300)
        self.fc1 = nn.Linear(embed_dim, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.out = nn.Linear(fc_dim2, 19)
        self.act = nn.RReLU()
        self.bn1 = nn.BatchNorm1d(fc_dim1)
        self.bn2 = nn.BatchNorm1d(fc_dim2)

    def forward(self, input):
        out = self.embed(input)

        u = F.tanh(self.proj(out))  # [batch, sentence_len, embed_dim]
        a = F.softmax(torch.einsum('bse,e->bs', (u.clone(), self.ctx_vec.clone())), dim=1)
        s = torch.einsum('bse,bs->be', (out.clone(), a.clone()))
        out = self.bn1(F.dropout(self.act(self.fc1(s)), p=0.5, training=self.training, inplace=True))
        out = self.bn2(F.dropout(self.act(self.fc2(out)), p=0.5, training=self.training, inplace=True))
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
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # seed = 1
    # model_fn = lambda: FastText(1024, 1024)
    # model_name = 'new_fasttext'
    # train_data = np.load('../../data/train_input.npy')
    # train_label = np.load('../../data/label.npy')
    # test_data = np.load('../../data/test_input.npy')
    # cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, seed=seed)

    seed = 1
    model_fn = lambda: Fast_Attention_Text(1024, 1024)
    model_name = 'fast_attention_text'
    train_data = np.load('../../data/train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/test_input.npy')
    # hold_out_test(model_fn, model_name, train_data, train_label, test_data)
    cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, batch_size=32, lr=5e-4,
                             seed=seed)
