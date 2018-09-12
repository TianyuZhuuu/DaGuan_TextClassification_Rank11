import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.dl_models.utils import cross_validation_bagging, hold_out_test

# Epoch 16 in 178.7s: improved!
#     train loss: 0.6304 valid loss: 0.7927
#     train f1_macro: 0.7983 valid f1_macro: 0.7609
# class TextGRUCNN(nn.Module):
#     def __init__(self, hidden_dim, num_filter, fc_dim):
#         super(TextGRUCNN, self).__init__()
#         embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
#         embed_dim = embed_mat.size(1)
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         self.embed = nn.Embedding.from_pretrained(embed_mat)
#         self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
#         self.conv = nn.ModuleList(
#             [nn.Conv2d(1, num_filter, (size, 2 * hidden_dim), bias=False) for size in range(1, 6)])
#         self.act = nn.PReLU()
#         self.fc = nn.Linear(5 * num_filter, fc_dim, bias=False)
#         self.out = nn.Linear(fc_dim, 19)
#         self.bn1 = nn.BatchNorm1d(5 * num_filter)
#         self.bn2 = nn.BatchNorm1d(fc_dim)
#
#         self._initialize_weights()
#
#     def forward(self, input):
#         out = F.dropout(self.embed(input), p=0.2, training=self.training, inplace=True)
#         out, _ = self.gru(out)  # [batch, sentence_len, 2*hidden_dim]
#         out = out.unsqueeze(dim=1)
#         out = [self.act(conv(out)).squeeze(3) for conv in self.conv]
#         out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
#         out = torch.cat(out, dim=1)
#         out = self.bn1(F.dropout(out, p=0.5, training=self.training, inplace=True))
#         out = self.bn2(F.dropout(self.act(self.fc(out)), p=0.2, training=self.training, inplace=True))
#         out = self.out(out)
#         return out
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.GRU):
#                 for name, param in m.named_parameters():
#                     if 'weight' in name:
#                         nn.init.orthogonal_(param)
#                     elif 'bias' in name:
#                         nn.init.constant_(param, 0.0)

class TextGRUCNN(nn.Module):
    def __init__(self, hidden_dim, num_filter, fc_dim):
        super(TextGRUCNN, self).__init__()
        embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
        embed_dim = embed_mat.size(1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embed = nn.Embedding.from_pretrained(embed_mat)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.ModuleList(
            [nn.Conv2d(1, num_filter, (size, 2 * hidden_dim), bias=False) for size in range(1, 6)])
        self.act = nn.PReLU()
        self.fc = nn.Linear(5 * num_filter, fc_dim, bias=False)
        self.out = nn.Linear(fc_dim, 19)
        self.bn1 = nn.BatchNorm1d(5 * num_filter)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self._initialize_weights()

    def forward(self, input):
        out = F.dropout(self.embed(input), p=0.2, training=self.training, inplace=True)
        out, _ = self.gru(out)  # [batch, sentence_len, 2*hidden_dim]
        out = out.unsqueeze(dim=1)
        out = [self.act(conv(out)).squeeze(3) for conv in self.conv]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, dim=1)
        out = self.bn1(F.dropout(out, p=0.5, training=self.training, inplace=True))
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
    model_fn = lambda: TextGRUCNN(40, 64, 128)
    model_name = 'textgrucnn'
    train_data = np.load('../../data/train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/test_input.npy')
    # hold_out_test(model_fn, model_name, train_data, train_label, test_data)
    cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, seed=2)
