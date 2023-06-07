import copy
import heapq
import pickle
import time
import matplotlib.pyplot as plt

import torch

from config import *

upoverhead = 0
downoverhead = 0
uptime = 0
downtime = 0
global_gradients = {}  # 全局梯度
global_gradient_accumulation = [0,0,0,0,0,0,0,0,0,0]  # 初始化累积的全局梯度
compensation = [0,0,0,0,0,0,0,0,0,0]  # 初始化补偿为0


# 第index个客户端更新
def train(index):
    l = 0
    for idx, (input, target) in enumerate(train_loader[index]):
        input = input.to(device)
        target = target.to(device)
        optimizer[index].zero_grad()
        output = models[index](input)
        loss = criterion(output, target)
        l = loss.item()
        loss.backward()
        optimizer[index].step()
    if PRINT:
        print(f"Client: {index} loss: {l:.2f}")
    return l


def train_first(index):
    l = 0
    for idx, (input, target) in enumerate(train_loader[index]):
        input = input.to(device)
        target = target.to(device)
        # optimizer[index].zero_grad() #梯度不置0
        output = models[index](input)
        loss = criterion(output, target)
        l = loss.item()
        # loss.backward() # 不在进行反向传播计算梯度
        optimizer[index].step()
    if PRINT:
        print(f"Client: {index} loss: {l:.2f}")


# 输入客户端编号也可以测试对应客户端的精度
# def pred(index):  #
#     with torch.no_grad():
#         total = 0
#         correct = 0
#         for idx, (input, target) in enumerate(test_loader):
#             input = input.to(device)
#             target = target.to(device)
#             output = models[index](input)
#             predict = output.argmax(1)
#             correct += predict.eq(target).sum().item()
#             total += len(input)
#         print(f"Accuracy:{correct / total}")


# 平均精度
def pred():  # 为了评估全局模型
    Accuracy = 0
    with torch.no_grad():
        for index in range(N_CLIENTS):
            total = 0
            correct = 0
            for idx, (input, target) in enumerate(test_loader):
                input = input.to(device)
                target = target.to(device)
                output = models[index](input)
                predict = output.argmax(1)
                correct += predict.eq(target).sum().item()
                total += len(input)
            Accuracy += (correct / total)*(1/N_CLIENTS)
    return Accuracy



def pred_clients(index):
    with torch.no_grad():
        total = 0
        correct = 0
        for idx, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = models[index](input)
            predict = output.argmax(1)
            correct += predict.eq(target).sum().item()
            total += len(input)
        Accuracy_client = correct / total
        print(f"Accuracy_client_{index+1}: {Accuracy_client:.2f}")




def pushup():  # 聚合梯度
    # global upoverhead, uptime
    # cur_overhead = 0
    # 平均梯度
    for name, param in models[1].named_parameters():  # 初始化全局梯度
        if param.grad is not None:
            global_gradients[name] = torch.zeros_like(models[1].state_dict()[name])
    for name in global_gradients:
            for idx in range(N_CLIENTS):
                name_params = dict(models[idx].named_parameters())  # 将迭代器转化为字典
                global_gradients[name] += (name_params[name].grad * (client_nums[idx] / total))

                # global_gradients[name] = pickle.dumps(global_gradients[name])   # 序列化为字节流
                # cur_overhead += len(global_gradients[name])    # 获取字节流的长度
                # global_gradients[name] = pickle.loads(global_gradients[name])   # 将字节流反序列化

    # upoverhead += cur_overhead  # 单次通信成本
    # cur_time = cur_overhead / (CLIENTBANDWIDTH * 1024 * N_CLIENTS)
    # uptime += cur_time
    # for _ in range(N_CLIENTS):
    #     while random.random() < CLIENTERROR:
    #         # 丢包重传
    #         uptime += cur_time
    # if TIME:
    #     time.sleep(cur_time)


def pushdown():  # 发送全局梯度
    # global downoverhead, downtime
    # cur_overhead = 0
    for name_1 in global_gradients:
        for idx in range(N_CLIENTS):
            for name_2, param in models[idx].named_parameters():
                if name_1 == name_2:
                    param.grad = global_gradients[name_1]
            #
            # tmp = pickle.dumps(global_gradients[name_1])  # 序列化为字节流
            # cur_overhead += len(global_gradients[name_1])  # 计算开销
            # global_gradients[name_1] = pickle.loads(tmp)  # 反序列化

    # cur_time = cur_overhead / (SERVERBANDWIDTH * 1024)
    # downoverhead += cur_overhead
    # downtime += cur_time
    # for _ in range(N_CLIENTS):
    #     while random.random() < SERVERERROR:
    #         downtime += cur_time / N_CLIENTS
    # if TIME:
    #     time.sleep(cur_time)


def convert_tensor_to_float(lst):  # 将列表中的tensor元素转化为float类型
    return [tensor.item() for tensor in lst]

def top_k_sparse_compensation(idx: int, global_epoch: int):  # 误差补偿topk
    gradient_list = []
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            gradient_list.extend(param.grad.view(-1))  # 转换梯度为一维列表
    gradient_list = convert_tensor_to_float(gradient_list)  # 将列表中的tensor转换为float
    gradient_tensor = torch.tensor(gradient_list)  # 转换成张量
    if global_epoch == 0:     # 第一轮先初始化误差为0
       compensation[idx] = torch.zeros_like(gradient_tensor)
    gradient_tensor = gradient_tensor + compensation[idx]   # 加入误差，
    values, indices = torch.topk(torch.abs(gradient_tensor), k)  # 计算绝对值最大的k个梯度及其索引
    sparse_gradients = torch.zeros_like(gradient_tensor)  # 创建一个与梯度张量形状相同的零张量
    sparse_gradients[indices] = gradient_tensor[indices]  # 将 top-k 值对应的梯度复制到零张量中
    compensation[idx] = gradient_tensor - sparse_gradients    # 计算误差值
    sparse_gradients = sparse_gradients.to(device)
    for param in models[idx].parameters():
        if param.grad is not None:
            param.grad.zero_()  # 将参数的梯度值全置为0
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            shape = param.grad.shape
            param.grad += sparse_gradients[:param.grad.numel()].view(shape)  # 切片选择对应层的梯度,并转换为与参数相同的size、
            sparse_gradients = sparse_gradients[param.grad.numel():]  # 去掉已经取出的部分



def top_k_sparse(idx: int):  # 普通topk
    gradient_list = []
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            gradient_list.extend(param.grad.view(-1))  # 转换梯度为一维列表
    gradient_list = convert_tensor_to_float(gradient_list)  # 将列表中的tensor转换为float
    gradient_tensor = torch.tensor(gradient_list)  # 转换成张量
    values, indices = torch.topk(torch.abs(gradient_tensor), k)  # 计算绝对值最大的k个梯度及其索引
    sparse_gradients = torch.zeros_like(gradient_tensor)  # 创建一个与梯度张量形状相同的零张量
    sparse_gradients[indices] = gradient_tensor[indices]  # 将 top-k 值对应的梯度复制到零张量中
    sparse_gradients = sparse_gradients.to(device)
    for param in models[idx].parameters():
        if param.grad is not None:
            param.grad.zero_()  # 将参数的梯度值全置为0
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            shape = param.grad.shape
            param.grad += sparse_gradients[:param.grad.numel()].view(shape)  # 切片选择对应层的梯度,并转换为与参数相同的size、
            sparse_gradients = sparse_gradients[param.grad.numel():]  # 去掉已经取出的部分


def hadamard_topk_sparse(idx: int):  # 点乘Top-K
    local_gradient_list = []
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            local_gradient_list.extend(param.grad.view(-1))  # 转换局部梯度为一维列表
    local_gradient_list = convert_tensor_to_float(local_gradient_list)  # 将列表中的tensor转化为float
    local_gradient_tensor = torch.tensor(local_gradient_list)  # 转换成张量
    local_gradient_tensor = local_gradient_tensor.to(device)   #放到GPU里
    global_gradient_tensor = torch.cat([tensor.view(-1) for tensor in global_gradients.values()])  # 转换全局梯度成张量

    if gradient_accumulation: # 是否累积
       decayed_gradient_tensor = decay_factor * global_gradient_tensor   # 乘以衰减因子
       global_gradient_accumulation[idx] += decayed_gradient_tensor   #  累计全局梯度
       hadamard_product = local_gradient_tensor * global_gradient_accumulation[idx]  # 计算全局梯度与局部梯度的哈达玛积
    else:
       hadamard_product = local_gradient_tensor * global_gradient_tensor  # 计算全局梯度与局部梯度的哈达玛积
    values, indices = torch.topk(hadamard_product, k)  # 根据哈达玛积得到索引值

    sparse_local_gradients = torch.zeros_like(local_gradient_tensor)   # 创建一个与梯度张量形状相同的零张量
    sparse_local_gradients[indices] = local_gradient_tensor[indices]  # 将索引对应的局部梯度中的梯度值复制到零张量中
    sparse_local_gradients = sparse_local_gradients.to(device)  # 放到GPU中

    for param in models[idx].parameters():
        if param.grad is not None:
            param.grad.zero_()  # 将参数的梯度值全置为0

    for name, param in models[idx].named_parameters():  # 将一维tensor转换为字典
        if param.grad is not None:
            shape = param.grad.shape
            param.grad += sparse_local_gradients[:param.grad.numel()].view(shape)  # 切片选择对应层的梯度,并转换为与参数相同的size、
            sparse_local_gradients = sparse_local_gradients[param.grad.numel():]  # 去掉已经取出的部分




def hadamard_topk_sparse_compensation(idx: int,global_epoch: int):  # 误差补偿点乘topk
    local_gradient_list = []
    for name, param in models[idx].named_parameters():
        if param.grad is not None:
            local_gradient_list.extend(param.grad.view(-1))  # 转换局部梯度为一维列表
    local_gradient_list = convert_tensor_to_float(local_gradient_list)  # 将列表中的tensor转化为float
    local_gradient_tensor = torch.tensor(local_gradient_list)  # 转换成张量
    local_gradient_tensor = local_gradient_tensor.to(device)   #放到GPU里
    if global_epoch == 1:  # 第二轮先初始化误差为0
        compensation[idx] = torch.zeros_like(local_gradient_tensor)
    local_gradient_tensor = local_gradient_tensor + compensation[idx]  # 加入误差，
    global_gradient_tensor = torch.cat([tensor.view(-1) for tensor in global_gradients.values()])  # 转换全局梯度成张量

    hadamard_product = local_gradient_tensor * global_gradient_tensor  # 计算全局梯度与局部梯度的哈达玛积
    values, indices = torch.topk(hadamard_product, k)  # 根据哈达玛积得到索引值

    sparse_local_gradients = torch.zeros_like(local_gradient_tensor)   # 创建一个与梯度张量形状相同的零张量
    sparse_local_gradients[indices] = local_gradient_tensor[indices]  # 将索引对应的局部梯度中的梯度值复制到零张量中
    sparse_local_gradients = sparse_local_gradients.to(device)  # 放到GPU中
    compensation[idx] = local_gradient_tensor - sparse_local_gradients  # 计算误差值

    for param in models[idx].parameters():
        if param.grad is not None:
            param.grad.zero_()  # 将参数的梯度值全置为0

    for name, param in models[idx].named_parameters():  # 将一维tensor转换为字典
        if param.grad is not None:
            shape = param.grad.shape
            param.grad += sparse_local_gradients[:param.grad.numel()].view(shape)  # 切片选择对应层的梯度,并转换为与参数相同的size、
            sparse_local_gradients = sparse_local_gradients[param.grad.numel():]  # 去掉已经取出的部分

def main():
    print("training start!")
    print(f"dataset: {args.dataset}   model:{args.model}   k: {k}   global_epoch: {EPOCH}  alpha: {ALPHA}  lr: {LR} 优化算法： {args.optim}")
    accuracy_values = []
    accuracy = 0
    loss_values = []
    l = 0
    for i in range(EPOCH):  # 全局迭代次数
        if i == 0:
            print("---------------------------------------------")
            print(f"global_epoch: 1 start!")
            global_loss_1 = 0
            for j in range(N_CLIENTS):
                for _ in range(LOCAL_E):
                    loss_1 = train(j)
                global_loss_1 += (loss_1 * (client_nums[j] / total)) # 计算全局损失
                # top_k_sparse(j)   # topk
                # top_k_sparse_compensation(j, i)  # 误差补偿topk
            print(f"global_loss: {global_loss_1:.2f}")  # 输出全局损失
            if PLOT:
                loss_values.append(loss_1)

        else:
            print("---------------------------------------------")
            print(f"global_epoch:{i + 1} start!")
            # print("global_gradients download")
            # 依次训练
            global_loss = 0
            for j in range(N_CLIENTS):
                for _ in range(LOCAL_E):  # 局部迭代次数
                    if _ == 0:   # 第一轮训练不使用反向传播的梯度更新模型
                        train_first(j)
                    else:        # 以后轮次使用全局梯度计算
                        loss = train(j)
                global_loss += (loss * (client_nums[j] / total) )  # 计算全局损失
                # top_k_sparse(j)   # topk
                # top_k_sparse_compensation(j, i)  # 误差补偿topk
                # hadamard_topk_sparse(j)   # 点乘topk
                hadamard_topk_sparse_compensation(j, i)  # 误差补偿点乘topk
            print(f"global_loss: {global_loss:.2f}")  # 输出全局损失
            if PLOT:
                loss_values.append(loss)
        accuracy = pred()  # 所有客户端的平均精度
        print(f"Accuracy: {accuracy:.2f}")
        # pred_clients(0)
        if PLOT:
            accuracy_values.append(accuracy)


        pushup()  # 聚合成全局梯度
        # pred(-1)  # 全局模型的测试精度
        # # print("local_gradients upload")
        # print(f"global_epoch:{i + 1} end!")
        pushdown()  # 拉取全局梯度
# 训练完成后画图
    if PLOT:
        epochs = list(range(1, len(loss_values) + 1))
        # 创建第一个图形对象和子图对象，用于绘制损失值曲线
        fig1, ax1 = plt.subplots()
        ax1.plot(epochs, loss_values)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        plt.savefig("训练损失.jpg")
        # 精度曲线
        fig2, ax2 = plt.subplots()
        ax2.plot(epochs, accuracy_values)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        plt.show()
        plt.savefig("测试精度.jpg")

    # global upoverhead, downoverhead, uptime, downtime
    # print(f"Total communication overhead:{upoverhead + downoverhead} bytes")
    # print(f"\tUp overhead {upoverhead} bytes")
    # print(f"\tDown overhead {downoverhead} bytes")
    # print(f"Total communication time:{round(uptime + downtime, 2)} seconds")  # 有点小问题，模拟的依次传输
    # print(f"\tUp time {round(uptime, 2)} seconds")
    # print(f"\tDown time {round(downtime, 2)} seconds")
