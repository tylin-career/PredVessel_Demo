import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from cseta.models.nn.helper import LinearZeroSlope, ReluPlusOne
from cseta.models.nn.model_config import ModelConfiguration


class DeepRNN(nn.Module):

    def __init__(self, config: ModelConfiguration, verbose=False):
        super(DeepRNN, self).__init__()

        self.config = config
        self.verbose = verbose
        self.n_seq = len(self.config.seq_features)
        self.n_con = len(self.config.con_features)
        self.n_cat = len(self.config.cat_features)
        if config.add_next_port_indicator:
            assert 'is_next_port' in config.seq_features
            assert 'seq_length' in config.con_features
            self.is_np_idx = config.seq_features.index('is_next_port')
            self.seq_length_idx = config.con_features.index('seq_length')

        self.linear = nn.Linear if not config.zero_rectifier_slope else LinearZeroSlope
        self.activation = self.config.activation
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.bn_seq = nn.BatchNorm1d(self.n_seq, momentum=None)
        self.bn_con = nn.BatchNorm1d(self.n_con, momentum=None)
        self.embd = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in zip(self.config.vars['n_class'], self.config.vars['embd_size'])]
        )
        self.lstm = nn.LSTM(input_size=self.n_seq,
                            hidden_size=self.config.lstm_hidden_size,
                            num_layers=self.config.lstm_num_layer, batch_first=True, dropout=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.n_seq, self.config.conv_out[0], k, stride=2, padding=int(k / 2))
                , self.activation(), nn.AvgPool1d(2)
                , nn.Conv1d(self.config.conv_out[0], self.config.conv_out[1], k, stride=2, padding=int(k / 2))
                , self.activation(), nn.AvgPool1d(2)
                , nn.Conv1d(self.config.conv_out[1], self.config.conv_out[2], k, stride=2, padding=int(k / 2))
                , self.activation(), nn.AvgPool1d(2)
            ) for k in self.config.conv_kernals
        ])

        fc_seq_in = self.config.conv_out[-1] * len(self.config.conv_kernals) + self.config.lstm_hidden_size
        fc_seq_out_0 = self.config.fc_seq_out[0]
        fc_seq_out_1 = 0 if self.config.PSA else self.config.fc_seq_out[1]
        self.fc_seq = nn.Sequential(
            self.linear(fc_seq_in, fc_seq_out_0), nn.BatchNorm1d(fc_seq_out_0), self.activation(),
            self.linear(fc_seq_out_0, fc_seq_out_1), nn.BatchNorm1d(fc_seq_out_1), self.activation()
        )

        fc_cat_out_0 = self.config.fc_cat_out[0]
        fc_cat_out_1 = self.config.fc_cat_out[1]
        self.fc_cat = nn.Sequential(
            self.dropout,
            self.linear(sum(self.config.vars['embd_size']), fc_cat_out_0), nn.BatchNorm1d(fc_cat_out_0), self.activation(), self.dropout,
            self.linear(fc_cat_out_0, fc_cat_out_1), nn.BatchNorm1d(fc_cat_out_1), self.activation(), self.dropout
        )

        fc_out_0 = self.config.fc_out[0]
        fc_out_1 = self.config.fc_out[1]
        fc_out_2 = self.config.fc_out[2]*2 if self.config.with_uncertainty else self.config.fc_out[2]
        self.fc = nn.Sequential(
            self.linear(fc_seq_out_1 + fc_cat_out_1 + self.n_con, fc_out_0), nn.BatchNorm1d(fc_out_0), self.activation(),
            self.linear(fc_out_0, fc_out_1), nn.BatchNorm1d(fc_out_1), self.activation(),
            self.linear(fc_out_1, fc_out_2)
            # self.linear(fc_out_1, fc_out_2)  #log transformation
        )
        self.relu_plus_one = ReluPlusOne(with_uncertainty=self.config.with_uncertainty, transform=self.config.target_transform)
        self.to(self.config.device)

    def add_next_port_indicator(self, x_seq, x):
        lengths = x[:, self.seq_length_idx].int()
        for i, len_np in enumerate(lengths):
            x_seq[i, -len_np:, self.is_np_idx] = 1
        return x_seq

    def forward(self, x_seq, x, device = 'cpu'):
        # self.lstm.flatten_parameters()
        if self.config.add_next_port_indicator:
            x_seq = self.add_next_port_indicator(x_seq, x)
        x_seq = x_seq.float().to(device)
        x_con = x[:, :self.n_con].float().to(device)
        x_cat = x[:, -self.n_cat:].long().to(device)
        current_bs = x.size(0)

        if self.config.batch_normalization:
            x_con = self.bn_con(x_con)
            x_seq = self.bn_seq(x_seq.permute(0, 2, 1))

        x_cat = [embd(x_cat[:, i]) for i, embd in enumerate(self.embd)]
        x_cat = torch.cat(x_cat, 1)
        x_cat = self.fc_cat(x_cat)
        print('embed:       ', x_cat.size()) if self.verbose else None

        if self.config.PSA:
            out = torch.cat([x_con, x_cat], dim=1)
        else:
            lstm_out, (hn, cn) = self.lstm(x_seq.permute(0, 2, 1))
            hn = hn[-1, :, :].view(current_bs, self.config.lstm_hidden_size)
            print('lstm hn:     ', hn.size()) if self.verbose else None

            cnn_out = [conv(x_seq) for conv in self.convs]  # conv layer with different kernals
            [print('   after conv:  ', i.size()) if self.verbose else None for i in cnn_out]
            cnn_out = torch.cat(cnn_out, dim=1)
            cnn_out = cnn_out.view(-1, cnn_out.size(1))

            out = torch.cat([hn, cnn_out], dim=1)
            out = self.fc_seq(out)
            out = torch.cat([out, x_con, x_cat], dim=1)

        print('concat:      ', out.size()) if self.verbose else None
        out = self.fc(out)
        out = self.relu_plus_one(out)
        print('output:      ', out.size()) if self.verbose else None
        return out



