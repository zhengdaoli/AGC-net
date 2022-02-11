from torch import optim

from util import *
from model import *
import matplotlib.pyplot as plt
import time
import os
from gwnn_model import *
from fastprogress import progress_bar
from exp_results import summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = Args()


def load_dataset(dataset_dir, batch_size):
    datasets = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if args.dev:
            datasets['x_' + category] = cat_data['x'][:args.dev_size]
        else:
            datasets['x_' + category] = cat_data['x']
        print(category + ' x size: ', datasets['x_' + category].shape)
        if args.dev:
            datasets['y_' + category] = cat_data['y'][:args.dev_size, ...]
        else:
            datasets['y_' + category] = cat_data['y']
        print(category + ' y size: ', datasets['y_' + category].shape)

    # normalization of first feature: speed
    scaler = StandardScaler(mean=datasets['x_train'][..., 0].mean(), std=datasets['x_train'][..., 0].std())

    # construct dataloader
    for category in ['train', 'val', 'test']:
        datasets['x_' + category][..., 0] = scaler.transform(datasets['x_' + category][..., 0])
        # construct data
        datasets[category + '_loader'] = TrafficDataLoader(datasets['x_' + category], datasets['y_' + category],
                                                           batch_size, args.cuda, transpose=args.transpose)
    print('finish load dataset!')
    return datasets, scaler


def rnn_train(args, datasets, scaler):
    _, _, adj_mx = load_adj(args.adj_file, 'normlap')
#     adj_mx[adj_mx > 0.01] = 1
#     adj_mx[adj_mx<=0.01] = 0
    print(adj_mx)
    
    if args.rnn:
        model = RNNModel(args)
        print('load rnnmodel')
    else:
        model = STWN(adj_mx[0], args, is_gpu=args.cuda)
        print('load STWN')
    if args.pretrain:
        print('pretrainok')
        model.load_state_dict(torch.load(args.pre_model_path))

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainer = Trainer(args, model, optimizer, scaler)

    best_model = dict()
    best_val_mae = 1000
    best_unchanged_threshold = 500  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()
    for e in range(args.epochs):
        print('Starting epoch: ', e)
        datasets['train_loader'].shuffle()
        train_loss, train_mae, train_mape, train_rmse = [], [], [], []
        for i, (input_data, target) in enumerate(datasets['train_loader'].get_iterator()):
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            # yspeed = target[:, 0, :, :]
            input_data, target = Variable(input_data), Variable(target)
            # mse, mae, mape, rmse = trainer.train(input_data, target)
            loss, mae, mape, rmse = trainer.train(input_data, target)
            # training metrics
            train_loss.append(loss)
            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

        # validation metrics
        # TODO: pick best model with best validation evaluation.
        datasets['val_loader'].shuffle()
        val_loss, val_mae, val_mape, val_rmse = [], [], [], []
        for _, (input_data, target) in enumerate(datasets['val_loader'].get_iterator()):
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()
            input_data, target = Variable(input_data), Variable(target)
            mae, mape, rmse = trainer.eval(input_data, target)

            # add metrics
            val_mae.append(mae)
            val_mape.append(mape)
            val_rmse.append(rmse)
            val_loss.append(mae)
        m = dict(train_loss=np.mean(train_loss), train_mae=np.mean(train_mae),
                 train_rmse=np.mean(train_rmse), train_mape=np.mean(train_mape),
                 valid_loss=np.mean(val_loss), valid_mae=np.mean(val_mae),
                 valid_mape=np.mean(val_mape), valid_rmse=np.mean(val_rmse)
                 )

        m = pd.Series(m)
        print(m)
        train_val_metrics.append(m)
        # once got best validation model ( 20 epochs unchanged), then we break.
        if m['valid_mae'] < best_val_mae:
            best_val_mae = m['valid_mae']
            best_count = 0
            best_model = trainer.model.state_dict()
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break
    #         trainer.scheduler.step()
    # test metrics
    torch.save(best_model, args.best_model_save_path)
    trainer.model.load_state_dict(torch.load(args.best_model_save_path))
    print('best_epoch:', best_index)

    test_metrics = []
    test_mae, test_mape, test_rmse = [], [], []
    for i, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
        input_data, target = Variable(input_data), Variable(target)
        if target.max() == 0: continue
        mae, mape, rmse = trainer.eval(input_data, target)
        # add metrics
        test_mae.append(mae)
        test_mape.append(mape)
        test_rmse.append(rmse)
    m = dict(test_mape=np.mean(test_mape), test_rmse=np.mean(test_rmse),
             test_mae=np.mean(test_mae))
    m = pd.Series(m)
    print("test:")
    print(m)
    
    test_metrics.append(m)
    plot(train_val_metrics, test_metrics, args.fig_filename)
    print('finish rnn_train!, time cost:', time.time() - start_time)
    # output learnable wavelets matrix:
    
    for i in range(len(model.gwblocks)):
        torch.save(model.gwblocks[i].wavelets, f"{i}_wavelets_maps.pt")
        
        


def plot(train_val_metrics, test_metrics, fig_filename='mae'):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_mae = [m['train_mae'] for m in train_val_metrics]
    val_mae = [m['valid_mae'] for m in train_val_metrics]

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_mae, '', label='train_mae')
    plt.plot(x, val_mae, '', label='val_mae')
    plt.title('MAE')
    plt.legend(loc='upper right')  # 设置label标记的显示位置
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.grid()

    plt.savefig(fig_filename)


#     plt.show()


if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = util.get_common_args()
    args.add_argument('--gwnnArgs', help='gwnnArgs')
    args.add_argument('--adj_mx', help='adj_mx')
    args = args.parse_args()

    print(args)
    datasets, scaler = load_dataset(args.data_path, args.batch_size)
    # t1 = time.time()

    rnn_train(args, datasets, scaler)
#     wavenet_train(args, datasets, scaler)
