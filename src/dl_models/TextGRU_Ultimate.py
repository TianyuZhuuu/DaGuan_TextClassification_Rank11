import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.dl_models.utils import hold_out_test, cross_validation_bagging


class TextGRU_Ultimate(nn.Module):
    def __init__(self, hidden_dim, fc_dim):
        super(TextGRU_Ultimate, self).__init__()
        embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
        embed_dim = embed_mat.size(1)
        self.hidden_dim = hidden_dim
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embed = nn.Embedding.from_pretrained(embed_mat)
        self.lstm = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True,
                           num_layers=2)
        self.ctx_vec_size = (2 * hidden_dim,)
        self.ctx_vec = nn.Parameter(torch.randn(self.ctx_vec_size).float().to(self.device))
        self.proj = nn.Linear(in_features=2 * hidden_dim, out_features=2 * hidden_dim)
        self.fc = nn.Linear(8 * hidden_dim, fc_dim, bias=False)
        self.act = nn.RReLU()
        self.out = nn.Linear(fc_dim, 19)
        self.bn1 = nn.BatchNorm1d(8 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self._initialize_weights()

    def forward(self, input):
        # input_shape = (input.size(0), input.size(1))
        # probs = torch.empty(input_shape).uniform_(0, 1).to(self.device)
        # spatial_dropout_input = torch.where(probs > 0.2, input,
        #                                     torch.zeros(input_shape, dtype=torch.int64).to(self.device))
        # del probs

        out = F.dropout(self.embed(input), p=0.2, training=self.training, inplace=True)
        out, _ = self.lstm(out)

        # last_pool = torch.cat((out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]), dim=1)
        last_pool = out[:, -1, :]
        max_pool, _ = torch.max(out, dim=1)
        avg_pool = torch.mean(out, dim=1)
        u = F.tanh(self.proj(out))  # [batch, sentence_len, 2*hidden_dim]
        a = F.softmax(torch.einsum('bsh,h->bs', (u.clone(), self.ctx_vec.clone())), dim=1)
        attention_pool = torch.einsum('bsh,bs->bh', (out.clone(), a.clone()))

        pool = torch.cat((last_pool, max_pool, avg_pool, attention_pool), dim=1)
        out = self.bn1(F.dropout(pool, p=0.5, training=self.training, inplace=True))
        out = self.bn2(F.dropout(self.act(self.fc(out)), p=0.2, training=self.training, inplace=True))
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
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)


if __name__ == '__main__':
    seed = 1
    model_fn = lambda: TextGRU_Ultimate(40, 128)
    model_name = 'textgru_ultimate'
    train_data = np.load('../../data/train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/test_input.npy')
    # hold_out_test(model_fn, model_name, train_data, train_label, test_data)
    cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, batch_size=64, seed=seed)
