import argparse
import pickle
import numpy as np

import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


def get_common_args():
    parser = argparse.ArgumentParser()
    # learning params
    parser.add_argument('--dev', action='store_true', help='dev')
    parser.add_argument('--dev_size', type=int, default=1000, help='dev_sample_size')
    parser.add_argument('--best_model_save_path', type=str, default='.best_model', help='best_model')
    parser.add_argument('--pre_model_path', type=str, default='./pre_model/best_model', help='pre_model_path')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='lr_decay_rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--clip', type=int, default=3, help='clip')
    parser.add_argument('--seq_length', type=int, default=3, help='seq_length')
    parser.add_argument('--predict_len', type=int, default=12, help='predict_len')
    parser.add_argument('--scheduler', action='store_true', help='scheduler')
    parser.add_argument('--mo', type=float, default=0.1, help='momentum')

    # running params
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--transpose', action='store_true', help='transpose sequence and feature?')
    parser.add_argument('--data_path', type=str, default='./data/METR-LA', help='data path')
    parser.add_argument('--adj_file', type=str, default='./data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adj_type', type=str, default='scalap', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--fig_filename', type=str, default='./mae', help='fig_filename')

    # model params
    parser.add_argument('--att', action='store_true', help='attention')
    parser.add_argument('--recur', action='store_true', help='recur')
    parser.add_argument('--fusion', action='store_true', help='fusion')
    parser.add_argument('--pretrain', action='store_true', help='pretrain')
    parser.add_argument('--feature_len', type=int, default=3, help='input feature_len')
    parser.add_argument('--gcn_layer_num', type=int, default=2, help='gcn_layer_num')
    parser.add_argument('--wavelets_num', type=int, default=20, help='wavelets_num')
    parser.add_argument('--rnn_layer_num', type=int, default=2, help='rnn_layer_num')
    parser.add_argument('--rnn_in_channel', type=int, default=32, help='rnn_in_channel')
    parser.add_argument('--rnn', action='store_true', help='attention')
    parser.add_argument('--decoder_type', type=str, default='rnn', help='decoder_type')


    return parser


# common parameters
class Args:
    def __init__(self,
                 dev=False,
                 data_path='./data/METR-LA',
                 best_model_save_path='./best_model',
                 batch_size=128,
                 epochs=10,
                 lr=0.0005,
                 weight_decay=0.0001,
                 cuda=False,
                 transpose=False,
                 params=dict(),
                 adj_type='scalap',
                 adj_file='',
                 seq_length=12,
                 n_iters=1,
                 clip=3,
                 lr_decay_rate=0.97,
                 addaptadj=True
                 ):
        self.dev = dev
        self.batch_size = batch_size
        self.best_model_save_path = best_model_save_path
        self.epochs = epochs
        self.lr = lr
        self.clip = clip
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.cuda = cuda
        self.transpose = transpose
        self.params = params
        self.adj_type = adj_type
        self.adj_file = adj_file
        self.seq_length = seq_length
        self.n_iters = n_iters
        self.addaptadj = addaptadj


class GWNNArgs:
    def __init__(self, num_nodes=207, do_grap_conv=True, p=True, aptonly=False, addaptadj=True, randomadj=False,
                 nhid=22, in_dim=2, dropout=0.3, apt_size=10, cat_feat_gc=False, clip=None):
        self.do_graph_conv = do_grap_conv
        self.p = p
        self.aptonly = aptonly
        self.addaptadj = addaptadj
        self.randomadj = randomadj
        self.nhid = nhid
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.dropout = dropout
        # self.n_obs = n_obs
        self.apt_size = apt_size
        self.cat_feat_gc = cat_feat_gc
        # self.fill_zeroes = fill_zeroes
        # self.checkpoint = checkpoint
        self.clip = clip
        self.lr_decay_rate = 0.97


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrafficDataLoader(object):
    def __init__(self, xs, ys, batch_size, cuda=False, transpose=False, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            # batch
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        xs = torch.Tensor(xs)
        ys = torch.Tensor(ys)
        if cuda:
            xs, ys = xs.cuda(), ys.cuda()
        # # temporal filter one sensor data as x, speed as y
        # xs = xs.squeeze()[:, :, [0]].squeeze()
        # ys = ys.squeeze()[:, :, [0]].squeeze()
        # ys = ys[:, :, 0]
        if transpose:
            xs = xs.transpose(1, 3)
            # ys = ys.transpose(1, 2)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            # TODO: Bug, we need to add more conditions.
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()

def norm(tensor_data, dim=0):
    mu = tensor_data.mean(axis=dim, keepdim=True)
    std = tensor_data.std(axis=dim, keepdim=True)
    return (tensor_data - mu) / (std + 0.00005)

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    print('origin adj:', adj)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def sym_norm_lap(adj):
    N = adj.shape[0]
    adj_norm = sym_adj(adj)
    L = np.eye(N) - adj_norm
    return L


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).toarray()
    normalized_laplacian = sp.eye(adj.shape[0]) - np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    
    adj_mx[adj_mx>0] = 1
    adj_mx[adj_mx<0] = 0
    
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32)]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj,


# TODO: to check how pytorch implement loss to avoid Nan.
def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    # handle all zeros.
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).cuda().transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def make_graph_inputs(args, device):
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_file, args.adj_type)
    if device == 'gpu':
        supports = [torch.tensor(i).cuda() for i in adj_mx]
    else:
        supports = [torch.tensor(i) for i in adj_mx]
    aptinit = None if args.gwnnArgs.randomadj else supports[
        0]  # ignored without do_graph_conv and add_apt_adj
    # if args.aptonly:
    #     if not args.addaptadj and args.do_graph_conv: raise ValueError(
    #         'WARNING: not using adjacency matrix')
    #     supports = None
    return aptinit, supports


def get_shared_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=40, help='Number of channels for internal conv')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--apt_size', default=10, type=int)
    parser.add_argument('--cat_feat_gc', action='store_true')
    parser.add_argument('--fill_zeroes', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='')
    return parser
