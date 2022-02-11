import argparse
import pickle
import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import linalg
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import util
from pygsp import graphs, filters
import numpy as np
import os
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def createGWCBlock(adj_mx):
    blocks = [STWNBlock(adj_mx, 2, 4, 4), STWNBlock(adj_mx, 4, 8, 4), STWNBlock(adj_mx, 8, 1, 4)]
    return blocks


def create_adj_kernel(N, size):
    Adj_kernel = nn.ParameterList(
        [nn.Parameter(torch.FloatTensor(N, N)) for _ in range(size)])
    return Adj_kernel


def create_mlp_kernel(in_channel, out_channel, kernel_size):
    mlp_kernel = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
    return mlp_kernel


def exp_wavelet_kernels(Lamda, scale):
    kernels = [np.exp(-Lamda[i] * scale) for i in range(len(Lamda))]
    # kernels = [np.exp(-Lamda[i] * 1.0 / i) for i in range(len(Lamda))]
    #     print('scale:',scale, kernels)
    return kernels


def create_wnn_kernel_by_pygsp(adj_mx):
    G = graphs.Graph(adj_mx)
    print('{} nodes, {} edges'.format(G.N, G.Ne))
    print(G)
    G.compute_laplacian('normalized')
    G.compute_fourier_basis()
    print('G.U', G.U)
    # print('G.U*G.UT', G.U )
    G.set_coordinates('ring2D')
    G.plot()


# TODO: this could be replaced by Chebyshev Polynomials
def create_wnn_kernel_matrix(norm_laplacian, scale):
    U, Lamda, _ = torch.svd(torch.from_numpy(norm_laplacian))
    kernels = exp_wavelet_kernels(Lamda, scale)
    # print(Lamda)
    G = torch.from_numpy(np.diag(kernels))
    Phi = np.matmul(np.matmul(U, G), U.t())
    Phi_inv = torch.inverse(Phi)
    # print('create_wnn_kernel_matrix: Phi:', Phi)
    return Phi, Phi_inv


class STWN(nn.Module):
    def __init__(self, adj_mx, args, is_gpu=False):
        super(STWN, self).__init__()
        self.is_gpu = is_gpu
        self.adj_mx = adj_mx
        self.N = adj_mx.shape[0]
        self.upsampling = nn.Conv2d(args.feature_len, 32, kernel_size=(1, 1))
        self.predict_len = args.predict_len
        self.args = args
        
        # add rnn encoder before gcn:
#         self.encoder = DecoderRNN(args.feature_len, 32, args.predict_len, num_layers=args.rnn_layer_num)
        self.encoder = EncoderRNN(args.feature_len, 32, args.predict_len,out_channel=args.predict_len, num_layers=args.rnn_layer_num)
        if args.att:
            print('using att')
            self.gwblocks = nn.ModuleList([AttSTWNBlock(adj_mx, args.feature_len, 32, args.wavelets_num, is_gpu=is_gpu)
                                           ])
            if args.gcn_layer_num > 1:
                for i in range(1, args.gcn_layer_num):
                    self.gwblocks.append(AttSTWNBlock(adj_mx, 32, 32, args.wavelets_num, is_gpu=is_gpu))
        else:
            print('no att', args.att)
            self.gwblocks = nn.ModuleList(
                [STWNBlock(adj_mx, 32, 32, args.wavelets_num, is_gpu=is_gpu)
                 ])
            if args.gcn_layer_num > 1:
                for i in range(1, args.gcn_layer_num):
                    self.gwblocks.append(STWNBlock(adj_mx, 32, 32, args.wavelets_num, is_gpu=is_gpu))
                print("gcn_layer_num: ", args.gcn_layer_num)
        # self.readout = STWNBlock(adj_mx, 4, 1, 4)
        # residual + input feature
        self.W_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))
        self.D_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))
        self.H_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))

        self.decoder = NewDecoderRNN(32, 32, self.predict_len, num_layers=args.rnn_layer_num)
#         self.decoder = DecoderRNN(32, 32, self.predict_len, num_layers=args.rnn_layer_num)

        self.lout = nn.Conv1d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, in_channel, N, sequence)
        """
        seq_len = x.shape[3] // 3
        w_x, d_x, h_x = x[:, :, :, :seq_len], x[:, :, :, seq_len:seq_len * 2], x[:, :, :, seq_len * 2:]

        if self.args.fusion:
            out, _ = self.forward_fusion(w_x, d_x, h_x)
        else:
            out, _ = self.forward_one(h_x)

        out = out.transpose(1, 2)
        return out, out

    def forward_fusion(self, w_x, d_x, h_x):
        # x = (B, F, N, T)
        # print(w_x.shape, d_x.shape, h_x.shape)
        out_w, _ = self.forward_one(w_x)
        out_d, _ = self.forward_one(d_x)
        out_h, _ = self.forward_one(h_x)
        # (batch, N, predict_len)
        out_w = torch.einsum("bnt,nt->bnt", out_w, self.W_W)
        out_d = torch.einsum("bnt,nt->bnt", out_d, self.D_W)
        out_h = torch.einsum("bnt,nt->bnt", out_h, self.H_W)

        out = out_w + out_d + out_h
        return out, out

    def forward_one(self, x):
        # x = (B, F, N, T)
        # encoder:
        
        # lstm + GCN + fc:
#         x = self.encoder(x)
        
        # GCN + lstm:
        residual = F.relu(self.upsampling(x))
        h = x
        for i in range(0, len(self.gwblocks)):
            h = residual + self.gwblocks[i](residual)

        # skip connection
        out = residual + h
        out = F.relu(out)

        # GCN + LSTM, decoder:
        out, h = self.decoder(out)
        
        # lstm + GCN + fc:
#         out = self.lout(out)

        # test without fc:
        # out = out.squeeze().transpose(1, 2)
        # out = self.fc(out)

        # out = B, N, T
        return out, out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                print('STWN init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    nn.init.normal_(i, mean=0.0, std=0.001)
                print('STWN init parameterlist with norm')
            else:
                print('STWN ParameterList!Do nothing')
        # nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')
        nn.init.normal_(self.W_W, mean=0.0, std=0.001)
        nn.init.normal_(self.D_W, mean=0.0, std=0.001)
        nn.init.normal_(self.H_W, mean=0.0, std=0.001)


def get_wavelet(kernel, scale, adj_mx, is_gpu):
    Phi, Phi_inv = create_wnn_kernel_matrix(adj_mx, scale)
    Phi = Phi.cuda()
    Phi_inv = Phi_inv.cuda()
    return Phi.mm(kernel.diag()).mm(Phi_inv)


class WaveletKernel(nn.Module):
    def __init__(self, adj_mx, is_gpu=False, scale=0.1):
        super(WaveletKernel, self).__init__()
        self.is_gpu = is_gpu
        self.adj_mx = adj_mx
        self.N = adj_mx.shape[0]
        self.scale = scale

        self.g = nn.Parameter(torch.ones(self.N))
        self.Phi, self.Phi_inv = create_wnn_kernel_matrix(self.adj_mx, self.scale)

        if is_gpu:
            self.g = nn.Parameter(torch.ones(self.N).cuda())
            self.Phi = self.Phi.cuda()
            self.Phi_inv = self.Phi_inv.cuda()
        g_diag = self.g.diag()
        self.k = self.Phi.mm(g_diag).mm(self.Phi_inv)
        self.weight_init()

    def forward(self, x):
        # batch, feature, N
        x = torch.einsum('bfn, np -> bfp', x, self.k).contiguous()
        return x

    def weight_init(self):
        # this could be replaced by Chebyshev Polynomials
        nn.init.uniform_(self.g)


# TODO: wavelet attention mechanism and trans attention
class STWNBlock(nn.Module):
    def __init__(self, adj_mx, in_channel, out_channel, wavelets_num, is_gpu=False):
        super(STWNBlock, self).__init__()
        self.is_gpu = is_gpu
        self.N = adj_mx.shape[0]
        self.adj_mx = adj_mx
        self.adj_mx_t = torch.from_numpy(adj_mx).float().cuda()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = wavelets_num

        scales = [0.1 + 0.1 * 2 * i for i in range(wavelets_num)]
        kernel_para = torch.ones(wavelets_num, self.N).float()
        if is_gpu:
            kernel_para = kernel_para.cuda()
        self.kernels = nn.ParameterList([nn.Parameter(kernel_para[i]) for i in range(wavelets_num)])
        self.wavelets = torch.stack(
            [get_wavelet(self.kernels[i], scales[i], self.adj_mx, is_gpu) for i in range(wavelets_num)],
            dim=0).cuda()
        self.randw = nn.Parameter(torch.randn(self.N, self.N).float()).cuda()
        self.krandw = nn.Parameter(torch.stack([torch.randn(self.N, self.N) for i in range(wavelets_num)], dim=0).float()).cuda()
#         self.wavelets = nn.Parameter(torch.randn(self.N, self.N).float())
        print('wavelets shape', self.wavelets.shape)
        self.Gate = nn.Parameter(torch.FloatTensor(wavelets_num)).cuda()
        self.SumOne = torch.ones(wavelets_num).float().cuda()
        self.upsampling = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
        self.rnn = nn.GRU(input_size=32,
                          hidden_size=32,
                          num_layers=1,
                          batch_first=True)
        self.weight_init()

        
    def forward(self, x):
        """
        :param x: (batch, in_channel, N, sequence)
        :return: (batch, out_channel, N, sequence)
        """
        
        seq_len = x.shape[3]
        seqs = []
        B, F, N, T = x.shape

        x = x.transpose(1, 2)

        
        # real wavelet + K randw
        wavelets = self.wavelets * self.krandw
        x = torch.einsum('bnft, knm -> bkmft', x, wavelets)
        x = torch.einsum('bknft, k -> bnft', x, self.Gate)


        # GCN + GRU:
        x = x.transpose(2, 3).reshape(B * N, T, F)
        outputs, last_hidden = self.rnn(x, None)
        outputs = outputs.reshape(B, N, T, F).permute(0, 3, 1, 2)
        return outputs

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                print('init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    nn.init.normal_(i, mean=0.0, std=0.001)
            else:
                print('ParameterList!Do nothing')
        nn.init.normal_(self.Gate, mean=0.0, std=0.001)
        # nn.init.kaiming_normal_(self.sampling)
        # nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')


class AttSTWNBlock(nn.Module):
    '''
    Attention STWNBlock
    '''

    def __init__(self, adj_mx, in_channel, out_channel, kernel_size, att_channel=32, bn=True, sampling=None,
                 is_gpu=False):
        super(AttSTWNBlock, self).__init__()
        self.is_gpu = is_gpu
        print('AttSTWNBlock, is_gpu', is_gpu)
        self.N = adj_mx.shape[0]
        self.adj_mx = adj_mx
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.att_channel = att_channel

        scales = [0.1 + 0.1 * 2 * i for i in range(kernel_size)]
        kernel_para = torch.ones(kernel_size, self.N).float()

        if is_gpu:
            kernel_para = kernel_para.cuda()

        self.kernels = nn.ParameterList([nn.Parameter(kernel_para[i]) for i in range(kernel_size)])
        self.wavelets = torch.stack(
            [get_wavelet(self.kernels[i], scales[i], self.adj_mx, is_gpu) for i in range(kernel_size)],
            dim=0)
        print('wavelets shape', self.wavelets.shape)
        if is_gpu:
            self.upsamplings = nn.Parameter(torch.FloatTensor(kernel_size, in_channel, out_channel).cuda())
        else:
            self.upsamplings = nn.Parameter(torch.FloatTensor(kernel_size, in_channel, out_channel))

        #         self.upsamplings = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channel, out_channel).cuda())
        #                                              for _ in range(self.kernel_size)])

        self.upsampling = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))

        self.Att_W = nn.Parameter(torch.FloatTensor(self.out_channel, self.att_channel))
        self.Att_U = nn.Parameter(torch.FloatTensor(kernel_size, self.att_channel))

        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, in_channel, N, sequence)
        :return: (batch, out_channel, N, sequence)
        """
        # TODO: change to recursive
        sha = x.shape
        seq_len = sha[3]
        seqs = []
        if self.is_gpu:
            wavelets = self.wavelets.cuda()
        else:
            wavelets = self.wavelets

        #         x = self.upsampling(x)
        #         print('samp',self.upsampling.weight)
        
        # attention:
        
        #       # wavelet gated, sum directly:
        x = torch.einsum('bnft, knm -> bkmft', x, wavelets)
        x = torch.einsum('bkmft, kfo -> bk')
        x = torch.einsum('bknft, k -> bnft', x, self.Gate)     
        
        
        a = torch.einsum('fs, bknf -> bkns', self.Att_W, xs)
        a = torch.einsum('bkns, ks -> bkn', xs, self.Att_U)
        
        for i in range(seq_len):
            xs = x[..., i].transpose(1, 2)
            # wavelet transform:
            # bnf, knn -> bknf
            xs = torch.einsum('bnf, ksn -> bknf', xs, wavelets)

            # in_channel to out_channel, bknf , kfs -> bkns
            xs = torch.einsum('bknf , kfo -> bkno', xs, self.upsamplings)

            mask = xs == float('-inf')
            xs = xs.data.masked_fill(mask, 0)
            # attention:

            # fs, bknf -> bkns
            # bkns, s -> bkn
            # bkn -> a = softmax(bkn, k)
            # bknf, bkn -> bknf
            a = torch.einsum('fs, bknf -> bkns', self.Att_W, xs)
            a = torch.einsum('bkns, ks -> bkn', xs, self.Att_U)
            a = util.norm(a, dim=1)
            a = F.softmax(a, dim=1)
            a = a.transpose(1, 2)

            # mock attention:
            #             xsshape = xs.shape
            #             a = torch.ones(xsshape[0], xsshape[2], xsshape[1]).float().cuda()

            # xs * attention
            out = torch.einsum('bnk, bkno -> bno', a, xs).transpose(1, 2)
            #             h = out
            seqs.append(out)

        # stack all sequences
        x = torch.stack(seqs, dim=3)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                print('init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    #                     nn.init.kaiming_normal_(i.data, mode='fan_out')
                    nn.init.normal_(i, mean=0.0, std=0.001)
                    # nn.init.kaiming_normal_(i.weight.data, mode='fan_in')
                print('init parameterlist with kaiming', m)
            else:
                print('ParameterList!Do nothing', m)
        # nn.init.kaiming_normal_(self.upsamplings.data, mode='fan_in')
        #         nn.init.normal_(self.upsamplings, mean=0.0, std=0.001)
        for m in self.upsamplings:
            nn.init.kaiming_normal_(m.data, mode='fan_in')

        nn.init.normal_(self.Att_W, mean=0.0, std=0.001)
        nn.init.kaiming_normal_(self.Att_U.data, mode='fan_in')


#         nn.init.normal_(self.Att_U, mean=0.0, std=0.001)


class DecoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, N, sequence, out_channel)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = out.transpose(1, 2)
            out = self.l1(out)
            seqs.append(out)

        predict = torch.cat(seqs, dim=2)
        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict



class NewDecoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(NewDecoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, N, sequence, out_channel)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len).transpose(1, 2)
        # last seq:
        x = x[:,-1,:].unsqueeze(dim=1)
        h = None
        seqs = []
        for i in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = out[:,-1,:].unsqueeze(dim=1)
            each_seq = out.transpose(1, 2)
            each_seq = self.l1(each_seq)
            seqs.append(each_seq)
        
        predict = torch.cat(seqs, dim=2).squeeze()
        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict

    
    
class EncoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, channel, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, in_channel, N, sequence)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len)
        h = None
        x = x.transpose(1, 2)
        seqs = []
        out, h = self.rnn(x, h)
        # batch*N, seq, feature.
        out = out.reshape(batch, N, seq_len, self.hidden_len)
        out = out[:,:,-1,:].unsqueeze(dim=2).permute(0, 3, 1, 2)
        return out


    
    
class DecoderLSTM(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, num_layers=1):
        super(DecoderLSTM, self).__init__()
        print('init DecoderLSTM')
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        # RNN层
        self.rnn = nn.LSTM(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        self.l1 = nn.Conv1d(hidden_len, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch,N,sequence)
        """
        batch, _, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, seq_len, self.feature_len).transpose(1, 2)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = self.l1(out.transpose(1, 2))
            seqs.append(out)

        predict = torch.cat(seqs, dim=1)

        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict


class DecoderGRU(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, num_layers=1):
        super(DecoderGRU, self).__init__()
        print('init DecoderGRU')
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        # RNN层
        self.rnn = nn.GRU(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        self.l1 = nn.Conv1d(hidden_len, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch,N,sequence)
        """
        batch, _, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, seq_len, self.feature_len).transpose(1, 2)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = self.l1(out.transpose(1, 2))
            seqs.append(out)

        predict = torch.cat(seqs, dim=1)

        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict


class FC(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(FC, self).__init__()
        self.N = N
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.l1 = nn.Conv2d(in_channel, 16, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(16)
        self.l2 = nn.Conv2d(16, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        """
        :param x: (batch, features, Nodes, sequence)
        :return: (batch, sequence, Nodes, features=1).squeeze() --> (batch, sequence, Nodes)
        """
        x = F.relu(self.bn(self.l1(x)))
        x = F.relu(self.l2(x))
        seq_len = x.shape[3]
        batch_size = x.shape[0]
        # outs = []
        # for i in range(self.N):
        #     node = x[:, :, i, :]
        #     out = F.relu(self.l(node))
        #     outs.append(out)
        x = x.reshape(batch_size, seq_len, self.N, self.out_channel).squeeze()
        # x = torch.cat(outs, dim=2).reshape(batch_size, seq_len, self.N, self.out_channel).squeeze()
        return x


class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        self.predict_len = args.predict_len
        self.args = args
        self.rnn_in_channel = args.rnn_in_channel
        self.upsampling = nn.Conv1d(args.feature_len, self.rnn_in_channel, kernel_size=1)
        if args.decoder_type == "rnn":
            self.decoder = DecoderRNN(self.rnn_in_channel, 32, self.predict_len, num_layers=args.rnn_layer_num)
        elif args.decoder_type == "lstm":
            print('lstm')
            self.decoder = DecoderLSTM(self.rnn_in_channel, 32, self.predict_len, num_layers=args.rnn_layer_num)
        else:
            self.decoder = DecoderGRU(self.rnn_in_channel, 32, self.predict_len, num_layers=args.rnn_layer_num)

    def forward(self, x):
        """
        :param x: (batch, feature, N, sequence)
        :return:
        """
        batch, feature, N, seq = x.shape
        seq = seq // 3
        _, _, x = x[:, :, :, :seq], x[:, :, :, seq:seq * 2], x[:, :, :, seq * 2:]

        x = x.transpose(1, 2).reshape(batch * N, feature, seq)
        x = F.relu(self.upsampling(x))
        x = x.reshape(batch, N, self.rnn_in_channel, seq).transpose(1, 2)
        x, _ = self.decoder(x)
        x = x.transpose(1, 2)
        return x, x


class Trainer:
    def __init__(self, args, model, optimizer, scaler, criterion=nn.MSELoss()):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = args.clip
        self.lr_decay_rate = args.lr_decay_rate
        self.epochs = args.epochs
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epochs: self.lr_decay_rate ** epochs)

    def train(self, input_data, target):
        self.model.train()
        self.optimizer.zero_grad()

        # train
        output, h = self.model(input_data)
        # print('output shape', output.shape)
        #         h.detach()

        # loss, weights update
        # TODO: squeeze performance?
        output = output.squeeze()
        # TODO: replace the inverse_transform function by self-define or in forward function.
        predict = self.scaler.inverse_transform(output)
        # TODO: compare speed of cal all metrics here or outside.
        # 1. cal metrics here
        # print('predict:', predict.shape)
        # print('target:', target.shape)

        # target [batch, N, seq]
        # loss = self.criterion(predict, target)
        mae, mape, rmse = util.calc_metrics(predict, target)
        mae.backward(retain_graph=True)
        # loss.backward()
        self.optimizer.step()
        return mae.item(), mae.item(), mape.item(), rmse.item()

    def eval(self, input_data, target):
        self.model.eval()

        output, h = self.model(input_data)  # [batch_size,seq_length,num_nodes]
        h.detach()

        output = output.squeeze()

        predict = self.scaler.inverse_transform(output)

        predict = torch.clamp(predict, min=0., max=70.)
        mae, mape, rmse = util.calc_metrics(predict, target)
        return mae.item(), mape.item(), rmse.item()
