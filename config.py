import argparse
import copy
import random

import numpy as np

from data import get_dataloader_train, get_dataloader_test
from model import *
from torch.nn import init

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)  # 随机数种子

#################################################
# 系统设置
#################################################
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)  # 学习率
parser.add_argument('--epoch', type=int, default=300)  # 全局epoch
parser.add_argument('--clients', type=int, default=10)  # 客户端的个数
parser.add_argument('--localepoch', type=int, default=3)  # 本地迭代轮次
# parser.add_argument('--mu', type=float, default=0)  # FedProx系数, 0则退化为FedAvg
parser.add_argument('--alpha', type=float, default=0.02)  # non-iid程度 越低代表不均匀程度越高
parser.add_argument('--print', type=bool, default=False)  # 是否打印客户端的训练情况
parser.add_argument('--data', type=bool, default=True)  # 输出数据在所有客户端的分布情况，以文件保存
parser.add_argument('--dataset', type=str, default="CIFAR10")  # MNIST或者CIFAR10
parser.add_argument('--model', type=str, default="CNN")  # MLP,CNN, VGG(11)
parser.add_argument('--time', type=bool, default=False)  # 是否模拟传输时间
# parser.add_argument('--clientbandwidth', type=float, default=100)  # 客户端传输速度，默认100MB/s
# parser.add_argument('--serverbandwidth', type=float, default=1000)  # 服务器传输速度，默认1000MB/s
# parser.add_argument('--clienterror', type=float, default=0.05)  # 客户端丢包率
# parser.add_argument('--servererror', type=float, default=0.01)  # 服务器丢包率
parser.add_argument('--device', type=str, default="cpu")  # 设备
parser.add_argument('--optim', type=str, default='sgd')  # 优化器，adam,sgd,qcsgd,scaffold
# parser.add_argument('--localgamma', type=float, default=5)  # qcsgd localgamma
# parser.add_argument('--globalgamma', type=float, default=15)  # qcsgd globalgamma
# parser.add_argument('--localeta', type=float, default=0.1)  # qcsgd localeta
# parser.add_argument('--globaleta', type=float, default=0.1)  # qcsgd globaleta
# parser.add_argument('--moon', type=bool, default=False)  # moon 有问题，不收敛
# parser.add_argument('--moonmu', type=float, default=False)  # moon mu
# parser.add_argument('--moont', type=float, default=False)  # moon temperature
parser.add_argument('--plot', type=bool, default=True)  # 是否绘制损失图和精度图

#################################################
# 通信压缩设置
#################################################
# parser.add_argument('--quanup', type=bool, default=False) # 是否量化上传的参数
# parser.add_argument('--quandown', type=bool, default=False)  # 是否量化下载的参数
# parser.add_argument('--quanbit', type=int, default=False)  # 量化比特位
# parser.add_argument('--sparseup', type=bool, default=False)  # 是否稀疏化上传的参数
# parser.add_argument('--sparsedown', type=bool, default=False)  # 是否稀疏化下载的参数，尽量不要稀疏化这部分
# parser.add_argument('--sparseratio', type=float, default=False)  # 稀疏化比例
# parser.add_argument('--compensation', type=float, default=0)  # 在上传过程中应用补偿
parser.add_argument('--k', type=int, default=10000)  # 稀疏化的k值
parser.add_argument('--gradient_accumulation',type=bool,default=False)  #  是否在本地累加全局梯度
parser.add_argument('--decay_factor',type=float,default=0.9)  #  衰减系数


args = parser.parse_args()

SEED = args.seed

BATCH_SIZE = args.batchsize
LR = args.lr
EPOCH = args.epoch
N_CLIENTS = args.clients
LOCAL_E = args.localepoch
# MU = args.mu
ALPHA = args.alpha
PRINT = args.print
DATA_DISTRIBUTION = args.data
DATASETNAME = args.dataset
TIME = args.time
# SERVERBANDWIDTH = args.serverbandwidth
# CLIENTBANDWIDTH = args.clientbandwidth
# SERVERERROR = args.servererror
# CLIENTERROR = args.clienterror
# LOCALGAMMA = args.localgamma
# GLOBALGAMMA = args.globalgamma
# LOCALETA = args.localeta
# GLOBALETA = args.globaleta
# MOON = args.moon
# MOONMU = args.moonmu
# MOONT = args.moont
PLOT=args.plot



# QUAN_UP = args.quanup
# QUAN_DOWN = args.quandown
# QUAN_BIT = args.quanbit
# SPARSE_UP = args.sparseup
# SPARSE_DOWN = args.sparsedown
# SPARSE_RATIO = args.sparseratio
# COMPENSATION = args.compensation
k = args.k
gradient_accumulation = args.gradient_accumulation
decay_factor = args.decay_factor

device = torch.device(args.device)  # 设备


# 设置随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(SEED)

# assert not (QUAN_UP and SPARSE_UP), "不可同时量化和稀疏化"
# assert not (QUAN_DOWN and SPARSE_DOWN), "不可同时量化和稀疏化"

# 最后一个代表全局模型
# 在这里修改模型
if DATASETNAME == "MNIST":
    if args.model == "CNN":
        models = [CNN_MNIST().to(device) for _ in range(N_CLIENTS + 1)]
    elif args.model == "MLP":
        models = [MLP_MNIST().to(device) for _ in range(N_CLIENTS + 1)]
    else:
        models = None
        assert models is not None, "暂无其他模型"
elif DATASETNAME == "CIFAR10":
    if args.model == "CNN":
        models = [CNN_CIFAR10().to(device) for _ in range(N_CLIENTS + 1)]
    elif args.model == "MLP":
        models = [MLP_CIFAR10().to(device) for _ in range(N_CLIENTS + 1)]
    elif args.model == "VGG":
        models = [vgg11().to(device) for _ in range(N_CLIENTS + 1)]
    else:
        models = None
        assert models is not None, "暂无其他模型"
else:
    models = None
    assert models is not None, "暂无其他数据集"

# if MOON:
#     models_copy = [None] * N_CLIENTS

# history_gradient = []  # 保存每个节点历史梯度信息
# for i in range(N_CLIENTS):
#     tmp = copy.deepcopy(models[i].state_dict())
#     for key in tmp:
#         init.zeros_(tmp[key])
#     history_gradient.append(tmp)
# 每个节点的优化器
if args.optim == 'adam':
    optimizer = [torch.optim.Adam(models[i].parameters(), lr=LR) for i in range(N_CLIENTS)]

elif args.optim == 'sgd':
    optimizer = [torch.optim.SGD(models[i].parameters(), lr=LR) for i in range(N_CLIENTS)]

elif args.optim == 'qcsgd':
    from optim import QCSGD
    optimizer = [QCSGD(models[i].parameters(), LOCALETA, LOCALGAMMA, device) for i in range(N_CLIENTS)]

elif args.optim == 'scaffold':
    from optim import ScafFold
    optimizer = [ScafFold(models[i].parameters(), LR) for i in range(N_CLIENTS)]
    local_c = [None] * N_CLIENTS

else:
    optimizer = None
    assert optimizer is not None, "暂无其他优化器"


# if COMPENSATION:
#     # assert QUAN_UP or SPARSE_UP, "无压缩，无需补偿"
#     # # 初始化补偿
#     # for i in range(N_CLIENTS):
#     #     tmp = copy.deepcopy(models[i].state_dict())
#     #     for key in tmp:
#     #         init.zeros_(tmp[key])
#     #     compensation.append(tmp)


# 数据
train_loader, client_nums, total = get_dataloader_train(BATCH_SIZE,
                                                        ALPHA,
                                                        N_CLIENTS,
                                                        DATA_DISTRIBUTION,
                                                        DATASETNAME,
                                                        device)

test_loader = get_dataloader_test(BATCH_SIZE, DATASETNAME)

# 交叉熵
criterion = torch.nn.CrossEntropyLoss()
