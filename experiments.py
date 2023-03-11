import gc
import shutil
import sys
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
import datetime
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/fedcnm/fedadam/fedadc/fedavgm')
    parser.add_argument('--comm_round', type=int, default=500, help='number of maximum communication roun')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=0, help="L2 regularization strength")
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--decay', type=float, default=1, help='learning rate decay per round')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer')
    parser.add_argument('--nag', type=bool, default=False, help='nesterov')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--alpha', type=float, default=0.4, help='The alpha parameter')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--tau', type=float, default=0.01, help='the tau parameter for fedadam')
    parser.add_argument('--glr', type=float, default=0.01, help='the global learning rate  for fedadam')
    args = parser.parse_args()
    return args

def init_nets(dropout_p, n_parties, args):
# def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16, 8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            if args.dataset == 'cifar100':
                net = ResNet18_cifar100()
            else:
                net = ResNet18_cifar10()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def train_net(net_id, train_dataloader, test_dataloader, epochs, lr, device="cpu"):
    # logger.info('Training network %s' % str(net_id))

    net = torch.load('./temp/g.pth').to(device)
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=args.reg)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg,nesterov=args.nag)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):#batch_size = 500/32 0-500/32
                x, target = x.to(device), target.to(device)
                #x = x.to(device)
                # target = target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True #tensor
                target.requires_grad = False
                target = target.long()

                out = net(x) #out=
                loss = criterion(out, target)

                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(),max_norm=1,norm_type=2)
                optimizer.step() #x^t=x^（t-1)-lr&delta
                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    torch.save(net, './temp/model' + str(net_id) + '.pth')
    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)

def local_train_net(selected,round, args, net_dataidx_map, test_dl=None, device="cpu"):
    for net_id in selected:
        gc.collect()
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,dataidxs)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        # n_epoch = args.epochs*(net_id%5+1)
        n_epoch = args.epochs*(net_id%5+1)
        lr = args.lr * (args.decay ** round)
        train_net(net_id, train_dl_local, test_dl, n_epoch, lr,  device=device)


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map



def train_fedcnm(round, net_id, k, train_dataloader, test_dataloader, epochs, args,  device="cpu"):
    lr = args.lr*(args.decay**round)
    nets, _, _ = init_nets( 0, 1, args)
    net = nets[0].to(device)
    # logger.info('Training network %s' % str(net_id))
    net_para= torch.load('./temp/g.pth')
    global_momentum = torch.load('./temp/global_momentum.pth')
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    for key in net_para:
        net_para[key] = net_para[key].to(float)
    for key in net_para:
        net_para[key] += lr * args.alpha * epochs * k * global_momentum[key].to(device)
        # net_para[key] += lr * alpha * epochs * k * global_momentum[key].to(device)
    net.load_state_dict(net_para)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=args.reg)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg,nesterov=args.nag)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(),max_norm=1,norm_type=2)
                # optimizer.step(net.reg_params)
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)
    torch.save(net, './temp/model' + str(net_id) + '.pth')


def local_train_fedcnm(round, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    for net_id in selected:
        gc.collect()
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        k = math.ceil(len(dataidxs) / args.batch_size)
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,dataidxs)
        n_epoch = args.epochs*(net_id%5+1)
        train_fedcnm(round, net_id, k, train_dl_local, test_dl, n_epoch, args, device=device)

def train_fedadc(round, net_id, k, train_dataloader, test_dataloader, epochs, args,  device="cpu"):
    lr = args.lr*(args.decay**round)
    # logger.info('Training network %s' % str(net_id))
    net = torch.load('./temp/g.pth').to(device)
    global_momentum = torch.load('./temp/global_momentum.pth')
    for key in global_momentum:
        global_momentum[key]=global_momentum[key].to(device)
        global_momentum[key]=global_momentum[key]/(k*epochs)
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            with torch.no_grad():
                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key].to(float)
                    net_para[key] += lr * global_momentum[key].to(device)
                net.load_state_dict(net_para)

            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(),max_norm=1,norm_type=2)
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)
    torch.save(net, './temp/model' + str(net_id) + '.pth')

def local_train_fedadc(round, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    for net_id in selected:
        gc.collect()
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        k = math.ceil(len(dataidxs) / args.batch_size)
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,dataidxs)
        n_epoch = args.epochs*(net_id%5+1)
        train_fedadc(round, net_id, k, train_dl_local, test_dl, n_epoch, args, device=device)


def train_fedprox(net_id, train_dataloader, test_dataloader, epochs, lr, device="cpu"):
    # logger.info('Training network %s' % str(net_id))
    nets, _, _ = init_nets( 0, 2, args)
    net = nets[0].to(device)
    net_para= torch.load('./temp/g.pth')
    net.load_state_dict(net_para)
    global_net = nets[1].to(device)
    global_net.load_state_dict(net_para)
    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=args.reg)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg,nesterov=args.nag)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    global_weight_collector = list(global_net.to(device).parameters())
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                #x = x.to(device)
                # target = target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True #tensor
                target.requires_grad = False
                target = target.long()

                out = net(x) #out=
                loss = criterion(out, target)

                fed_prox_reg = 0.0
                for param_index,param in enumerate(net.parameters()):
                    fed_prox_reg += ((args.mu/2)*torch.norm((param-global_weight_collector[param_index]))**2)

                loss+=fed_prox_reg
                loss.backward()
                nn.utils.clip_grad_norm(net.parameters(),max_norm=1,norm_type=2)
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    torch.save(net, './temp/model' + str(net_id) + '.pth')
    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)

def local_train_fedprox(selected,round, args, net_dataidx_map, test_dl=None, device="cpu"):
    for net_id in selected:
        gc.collect()
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,dataidxs)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        # n_epoch = args.epochs*(net_id%5+1)
        n_epoch = args.epochs*(net_id%5+1)
        lr = args.lr * (args.decay ** round)
        train_fedprox(net_id, train_dl_local, test_dl, n_epoch, lr,  device=device)


if __name__ == '__main__':

    log_file_name = '{alg}_{time}'
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    mkdirs('./temp')
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path = log_file_name.format(alg=args.alg,time=datetime.datetime.now().strftime("%m-%d_%H-%M")) + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    logger.info(args)
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)
    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        global_models, _,_= init_nets(0, 1, args)
        global_model = global_models[0].to(device)
        global_para = global_model.state_dict()
        torch.save(global_model, './temp/g.pth')
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            local_train_net(selected,round, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
            with torch.no_grad():
                global_para = global_model.state_dict()
                new_global_para = copy.deepcopy(global_model.state_dict())
                for idx in range(len(selected)):
                    net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                    net_para = net.state_dict()
                    if idx == 0:
                        for key in net_para:
                            new_global_para[key] = (net_para[key] - global_para[key])  * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            new_global_para[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                for key in global_para:
                    global_para[key] = global_para[key].to(float)
                    global_para[key] += new_global_para[key]
                global_model.load_state_dict(global_para)
                torch.save(global_model, './temp/g.pth')
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc= compute_accuracy(global_model, test_dl_global,device=device)
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        global_models, _,_= init_nets(0, 1, args)
        global_model = global_models[0].to(device)
        global_para = global_model.state_dict()
        torch.save(global_para, './temp/g.pth')
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            local_train_fedprox(selected,round, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
            with torch.no_grad():
                global_para = global_model.state_dict()
                new_global_para = copy.deepcopy(global_model.state_dict())
                for idx in range(len(selected)):
                    net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                    net_para = net.state_dict()
                    if idx == 0:
                        for key in net_para:
                            new_global_para[key] = (net_para[key] - global_para[key])  * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            new_global_para[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                for key in global_para:
                    global_para[key] = global_para[key].to(float)
                    global_para[key] += new_global_para[key]
                global_model.load_state_dict(global_para)
                torch.save(global_para, './temp/g.pth')
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc= compute_accuracy(global_model, test_dl_global,device=device)
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedcnm':
        global_models, global_model_meta_data, global_layer_type = init_nets( 0, 1, args)
        global_model = global_models[0].to(device)
        global_para = global_model.state_dict()
        torch.save(global_para, './temp/g.pth')
        global_momentum =copy.deepcopy(global_para)
        for key in global_momentum:
            global_momentum[key] *= 0
        torch.save(global_momentum, './temp/global_momentum.pth')
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            local_train_fedcnm(round, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)

            with torch.no_grad():
                total_data_points = sum([len(net_dataidx_map[r]*(r%5+1)) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]*(r%5+1)) / total_data_points for r in selected]
                global_para = global_model.state_dict()
                delta = copy.deepcopy(global_para)
                sum_k=0
                for idx in range(len(selected)):
                    net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                    net_para = net.state_dict()
                    if idx == 0:
                        k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size)*args.epochs*(selected[idx]%5+1)
                        sum_k+=k
                        for key in net_para:
                            delta[key] = (net_para[key]-global_para[key])/(k*args.lr) * fed_avg_freqs[idx]
                            # new_global_para[key] = (net_para[key] - global_para[key])  * fed_avg_freqs[idx]
                            # new_global_para[key] = (net_para[key] - global_para[key]) / (k * args.lr)
                    else:
                        k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs*(selected[idx]%5+1)
                        sum_k+=k
                        for key in net_para:
                            delta[key] += (net_para[key]-global_para[key])/(k*args.lr) * fed_avg_freqs[idx]
                            # new_global_para[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                            # new_global_para[key] += (net_para[key] - global_para[key]) / (k * args.lr)
                torch.save(delta,'./temp/global_momentum.pth')
                sum_k/=len(selected)
                for key in global_para:
                    global_para[key] = global_para[key].to(float)
                    global_para[key]+=delta[key]*sum_k*args.lr
                    # global_para[key] += new_global_para[key]
                global_model.load_state_dict(global_para)
                torch.save(global_para, './temp/g.pth')

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl_global))

            train_acc = compute_accuracy(global_model, train_dl_global,device = device)
            test_acc= compute_accuracy(global_model, test_dl_global, device = device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedavgm':
        logger.info("Initializing nets")
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0].to(device)

        global_para = global_model.state_dict()

        momentum = copy.deepcopy(global_model.state_dict())
        torch.save(momentum, './temp/momentum.pth')
        torch.save(global_model, './temp/g.pth')

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            local_train_net(selected, round, args, net_dataidx_map, test_dl=test_dl_global, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            global_para = global_model.state_dict()
            new_global_para = copy.deepcopy(global_para)
            sum_k = 0
            for idx in range(len(selected)):
                net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                net_para = net.state_dict()
                if idx == 0:
                    # k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs
                    for key in net_para:
                        # new_global_para[key] = (net_para[key] - global_para[key]) / (k * args.lr) * fed_avg_freqs[idx]
                        new_global_para[key] = (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                        # new_global_para[key] = (net_para[key] - global_para[key])
                else:
                    # k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs
                    for key in net_para:
                        # new_global_para[key] += (net_para[key] - global_para[key]) / (k * args.lr) * fed_avg_freqs[idx]
                        new_global_para[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                        # new_global_para[key] += (net_para[key] - global_para[key])
            # torch.save(new_global_para, './temp/delta.pth')

            momentum = torch.load('./temp/momentum.pth')
            if round==0:
                for key in momentum:
                    momentum[key] = new_global_para[key]
            else:
                for key in momentum:
                    momentum[key] = args.alpha*momentum[key]+new_global_para[key]
            torch.save(momentum, './temp/momentum.pth')
            for key in global_para:
                global_para[key]=global_para[key].to(float)
                global_para[key] += momentum[key]

            global_model.load_state_dict(global_para)
            torch.save(global_model, './temp/g.pth')

            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc = compute_accuracy(global_model, test_dl_global,device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedadam':
        logger.info("Initializing nets")
        global_models, global_model_meta_data, global_layer_type = init_nets( 0, 1, args)
        global_model = global_models[0].to(device)
        global_para = global_model.state_dict()

        momentum1 = copy.deepcopy(global_model.state_dict())
        momentum2 = copy.deepcopy(global_model.state_dict())
        for key in momentum1:
            momentum1[key] *= 0
            momentum2[key] *= 0
        torch.save(momentum1, './temp/momentum1.pth')
        torch.save(momentum2, './temp/momentum2.pth')
        torch.save(global_model, './temp/g.pth')

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            local_train_net(selected,round, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            with torch.no_grad():
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                global_para = global_model.state_dict()
                delta = torch.load('./temp/momentum1.pth') #delta
                sum_k = 0
                for idx in range(len(selected)):
                    net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                    net_para = net.state_dict()
                    if idx == 0:
                        # k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs
                        for key in net_para:
                            # new_global_para[key] = (net_para[key] - global_para[key]) / (k * args.lr) * fed_avg_freqs[idx]
                            delta[key] = (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                            # new_global_para[key] = (net_para[key] - global_para[key])
                    else:
                        # k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs
                        for key in net_para:
                            # new_global_para[key] += (net_para[key] - global_para[key]) / (k * args.lr) * fed_avg_freqs[idx]
                            delta[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                            # new_global_para[key] += (net_para[key] - global_para[key])
                # torch.save(new_global_para, './temp/delta.pth')

                momentum1 = torch.load('./temp/momentum1.pth')
                momentum2 = torch.load('./temp/momentum2.pth')
                if round==0:
                    for key in momentum1:
                        momentum1[key] = delta[key]
                        momentum2[key] = delta[key]**2
                else:
                    for key in momentum1:
                        momentum1[key] = args.alpha*momentum1[key]+(1-args.alpha)*delta[key]
                        momentum2[key] = args.mu*momentum2[key]+(1-args.mu)*(delta[key]**2)

                torch.save(momentum1, './temp/momentum1.pth')
                torch.save(momentum2, './temp/momentum2.pth')
                for key in global_para:
                    # global_para[key] += new_global_para[key]/ len(selected)
                    global_para[key] = global_para[key].to(float)
                    global_para[key] += args.glr*(momentum1[key]/(torch.sqrt(momentum2[key])+args.tau))
                    # logger.info(torch.sqrt(momentum2[key]))
                    # global_para[key] += args.glr*(momentum1[key]/(args.tau))
                global_model.load_state_dict(global_para)

                torch.save(global_model, './temp/g.pth')

                train_acc = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc = compute_accuracy(global_model, test_dl_global, device=device)

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedadc':
        global_models, _, _= init_nets( 0, 1, args)
        global_model = global_models[0].to(device)
        torch.save(global_model, './temp/g.pth')
        global_para = global_model.state_dict()
        global_momentum =copy.deepcopy(global_para)
        for key in global_momentum:
            global_momentum[key] *= 0
        torch.save(global_momentum, './temp/global_momentum.pth')
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            local_train_fedadc(round, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)

            with torch.no_grad():
                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
                global_para = global_model.state_dict()
                delta = copy.deepcopy(global_para)
                sum_k=0
                lr = args.lr * (args.decay ** round)
                for idx in range(len(selected)):
                    net = torch.load('./temp/model' + str(selected[idx]) + '.pth')
                    net_para = net.state_dict()
                    if idx == 0:
                        k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size)*args.epochs*(selected[idx]%5+1)
                        sum_k+=k
                        for key in net_para:
                            delta[key] = (net_para[key]-global_para[key]) * fed_avg_freqs[idx]/lr
                            # new_global_para[key] = (net_para[key] - global_para[key])  * fed_avg_freqs[idx]
                            # new_global_para[key] = (net_para[key] - global_para[key]) / (k * args.lr)
                    else:
                        k = math.ceil(len(net_dataidx_map[selected[idx]]) / args.batch_size) * args.epochs*(selected[idx]%5+1)
                        sum_k+=k
                        for key in net_para:
                            delta[key] += (net_para[key]-global_para[key]) * fed_avg_freqs[idx]/lr
                            # new_global_para[key] += (net_para[key] - global_para[key]) * fed_avg_freqs[idx]
                            # new_global_para[key] += (net_para[key] - global_para[key]) / (k * args.lr)
                global_momentum = torch.load('./temp/global_momentum.pth')
                for key in global_momentum:
                    global_momentum[key]=delta[key]+(1-args.mu)*global_momentum[key]
                torch.save(global_momentum,'./temp/global_momentum.pth')
                sum_k/=len(selected)
                for key in global_para:
                    global_para[key] = global_para[key].to(float)
                    global_para[key]+=global_momentum[key]*lr*args.alpha
                    # global_para[key] += new_global_para[key]
                global_model.load_state_dict(global_para)
                torch.save(global_model, './temp/g.pth')

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl_global))

            train_acc = compute_accuracy(global_model, train_dl_global,device = device)
            test_acc= compute_accuracy(global_model, test_dl_global, device = device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    shutil.rmtree('./temp')