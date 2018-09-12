import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.dl_models.utils import cross_validation_bagging


class Pooled_BiLSTM(nn.Module):
    def __init__(self, hidden_dim, fc_dim):
        super(Pooled_BiLSTM, self).__init__()
        embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
        embed_dim = embed_mat.size(1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embed = nn.Embedding.from_pretrained(embed_mat)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4 * hidden_dim, fc_dim)
        self.out = nn.Linear(fc_dim, 19)
        self.bn1 = nn.BatchNorm1d(4 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self._initialize_weights()

    def forward(self, input):
        out = F.dropout(self.embed(input), p=0.2, training=self.training, inplace=True)
        out, _ = self.lstm(out)
        max_pool, _ = torch.max(out, dim=1)
        avg_pool = torch.mean(out, dim=1)
        out = self.bn1(F.dropout(torch.cat((max_pool, avg_pool), dim=1), p=0.3, training=self.training, inplace=True))
        out = self.bn2(F.dropout(self.fc(out), p=0.5, training=self.training, inplace=True))
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)


if __name__ == '__main__':
    model_fn = lambda: Pooled_BiLSTM(40, 128)
    model_name = 'pooled_bilstm'
    train_data = np.load('../../data/train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/test_input.npy')
    cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data)
