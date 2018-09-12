import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.dl_models.utils import cross_validation_bagging, hold_out_test


#
# # 10-fold CV: 0.7602 LB: 0.
# class TextCNN(nn.Module):
#     def __init__(self, filter_sizes, num_filter, fc_dim1):
#         super(TextCNN, self).__init__()
#         embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
#         num_word, embed_dim = embed_mat.size()
#         self.embed = nn.Embedding.from_pretrained(embed_mat, freeze=False)
#         self.conv = nn.ModuleList([nn.Conv2d(1, num_filter, (size, embed_dim), bias=False) for size in filter_sizes])
#         self.act = nn.LeakyReLU()
#         self.word_dropout = nn.Dropout(0.1)
#         self.concat_dropout = nn.Dropout(0.2)
#         self.fc_dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(len(filter_sizes) * num_filter, fc_dim1)
#         self.out = nn.Linear(fc_dim1, 19)
#         self.bn1 = nn.BatchNorm1d(len(filter_sizes) * num_filter)
#         self.bn2 = nn.BatchNorm1d(fc_dim1)
#         self._initialize_weights()
#
#     def forward(self, input):
#         embed_out = F.dropout(self.embed(input), p=0.1, training=self.training, inplace=True)
#         embed_out = embed_out.unsqueeze(1)
#         conv_out = [self.act(conv(embed_out)).squeeze(3) for conv in self.conv]
#         conv_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]
#         conv_out = torch.cat(conv_out, dim=1)
#         out = self.bn1(F.dropout(conv_out, p=0.2, training=self.training, inplace=True))
#         fc_out = self.bn2(F.dropout(self.act(self.fc1(out)), p=0.5, training=self.training, inplace=True))
#         out = self.out(fc_out)
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
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)


# 10-fold CV: 0.7602 LB: 0.
class TextCNN(nn.Module):
    def __init__(self, filter_sizes, num_filter, fc_dim1):
        super(TextCNN, self).__init__()
        embed_mat = torch.from_numpy(np.load('../../data/word_embed_mat.npy').astype(np.float32))
        num_word, embed_dim = embed_mat.size()
        self.embed = nn.Embedding.from_pretrained(embed_mat, freeze=False)
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filter, (size, embed_dim), bias=False) for size in filter_sizes])
        self.act = nn.RReLU()
        self.fc = nn.Linear(len(filter_sizes) * num_filter, fc_dim1)
        self.out = nn.Linear(fc_dim1, 19)
        self.bn1 = nn.BatchNorm1d(len(filter_sizes) * num_filter)
        self.bn2 = nn.BatchNorm1d(fc_dim1)
        self._initialize_weights()

    def forward(self, input):
        embed_out = F.dropout(self.embed(input), p=0.1, training=self.training, inplace=True)
        embed_out = embed_out.unsqueeze(1)
        conv_out = [self.act(conv(embed_out)).squeeze(3) for conv in self.conv]
        conv_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]
        conv_out = torch.cat(conv_out, dim=1)
        out = self.bn1(F.dropout(conv_out, p=0.5, training=self.training, inplace=True))
        fc_out = self.bn2(F.dropout(self.act(self.fc(out)), p=0.25, training=self.training, inplace=True))
        out = self.out(fc_out)
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
    seed = 1
    # for seed in range(5):
    model_fn = lambda: TextCNN([2, 3, 4, 5], 128, 1024)
    model_name = 'textcnn_new'
    train_data = np.load('../../data/train_input.npy')
    train_label = np.load('../../data/label.npy')
    test_data = np.load('../../data/test_input.npy')
    hold_out_test(model_fn, model_name, train_data, train_label, test_data, batch_size=64, lr=5e-4)
    # cross_validation_bagging(model_fn, model_name, train_data, train_label, test_data, batch_size=32, lr=5e-4,
    #                          seed=seed)
