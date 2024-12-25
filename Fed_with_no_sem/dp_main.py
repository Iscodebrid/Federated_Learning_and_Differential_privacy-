'''
@Project ：GETNext-master 
@File    ：dp_main.py
@IDE     ：PyCharm 
@Author  ：Yangjie
@Date    ：2024/3/11 13:16 
@清华源   ：-i https://pypi.tuna.tsinghua.edu.cn/simple
'''
import pdb
import random
from collections import OrderedDict
from math import exp
from typing import Dict, Tuple

import numpy as np
from flwr.common import Scalar, NDArrays
from statsmodels.sandbox.tsa.diffusion import Diffusion
from torch import optim
from gaussian_moments import moments_calcu
from layers import *
import torch
import torch.nn as nn
import flwr as fl

PARAMS = {
    "batch_size": 1,
}


class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, dropout=0.1, timesteps=1000):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)
        self.SelfAttn = SelfAttn(embed_dim, embed_dim)
        self.Embed = Embed(ex, embed_dim, l_dim - 1, embed_layers)
        self.Attn = Attn(emb_l, l_dim - 1)
        self.diffusion = DiffusionModel(embed_dim, timesteps=timesteps)

    def forward(self, traj, mat1, mat2, vec, traj_len):
        # long(N, M, [u, l, t]), float(N, M, M, 2), float(L, L), float(N, M), long(N)
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)  # (N, M, emb), (N, M, M, emb)
        self_attn = self.SelfAttn(joint, delta, traj_len)  # (N, M, emb)

        sde_output = self.sde.reverse_sde(self_attn, joint, t=0.1)

        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len)  # (N, M, L, emb)
        # output = self.Attn(self_attn, self_delta, traj_len)  # (N, L)
        output = self.Attn(sde_output, self_delta, traj_len)

        t = torch.randint(0, self.diffusion.timesteps, (traj.size(0),))
        pred_noise = self.diffusion(output, t)

        return output, pred_noise


def calculate_acc(prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            # topk_predict (k)
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)


def calculate_mrr(prob, label):
    # prob (N, L), label (N)
    num_samples = label.size(0)
    mrr = 0.0

    # 将 prob 和 label 转换为 numpy 数组
    prob_npy = to_npy(prob)
    label_npy = to_npy(label)

    for i in range(num_samples):
        # 获取每个样本的排序索引
        sorted_indices = np.argsort(prob_npy[i])[::-1]
        # 找到第一个正确答案的排名
        rank = np.where(sorted_indices == label_npy[i])[0][0] + 1
        # 累加倒数排名
        mrr += 1.0 / rank

    # 计算平均倒数排名
    mrr /= num_samples

    return mrr


def calculate_precision(prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    precision = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            # topk_predict (k)
            if to_npy(label)[j] in topk_predict:
                precision[i] += 1

        # precision = TP / (TP + FP)
        precision[i] /= k

    return np.array(precision)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1] - 1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label - 1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg + len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m + 1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m + 1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i - len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


def train(model, trainloader, mat2s, cid, optimizer, config):
    model.to(device)
    num_neg = 10  # n neg
    print(f"  ---client_{cid}--start train---  ")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1)
    loss_t = []
    test_acc = []
    test_Mrr = []
    test_Acc = []
    pre_grad_l2norms = {}

    if config["enable_avg"]:
        epsilon = config["epsilon"] / config["num_round"]
        epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
    elif config["enable_seq"]:
        epsilon = 1 + config["num_round"]*(1/3)    # an = a1+n*(1/3)
        epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
    elif config["enable_ser"]:
        epsilon = config["epsilon"] / config["num_round"]
        epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)

    elif config["server_round"] == 1:
        epsilon = 2
        epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
    else:
        epsilon = 2 * sqrt((2 / 2) ** 2 + ((18 * config["server_round"]) / 40) ** 2)
        epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)

    max_test_acc = [0, 0, 0, 0]
    min_loss_test = float('inf')
    max_Mrr = 0.0
    max_test_Acc = [0, 0, 0, 0]

    for t in range(1, epochs + 1):
        train_loss = []
        cum_test = [0, 0, 0, 0]
        cum_ACC = [0, 0, 0, 0]
        cum_Mrr = []
        test_size = 0
        for step, item in enumerate(trainloader):
            # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
            person_input, person_m1, person_m2t, person_label, person_traj_len = item

            # first, try batch_size = 1 and mini_batch = 1
            # pdb.set_trace()
            input_mask = torch.zeros((PARAMS["batch_size"], max_len, 3), dtype=torch.long).to(device)
            m1_mask = torch.zeros((PARAMS["batch_size"], max_len, max_len, 2), dtype=torch.float32).to(device)

            for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
                input_mask[:, :mask_len] = 1.
                m1_mask[:, :mask_len, :mask_len] = 1.

                train_input = person_input * input_mask
                train_m1 = person_m1 * m1_mask
                train_m2t = person_m2t[:, mask_len - 1]
                train_label = person_label[:, mask_len - 1]  # (N)
                train_len = torch.zeros(size=(PARAMS["batch_size"],), dtype=torch.long).to(device) + mask_len

                prob = model(train_input, train_m1, mat2s, train_m2t, train_len)  # (N, L)

                if mask_len <= person_traj_len[0] * 2 / 3:
                    prob_sample, label_sample = sampling_prob(prob, train_label, num_neg)
                    loss_train = F.cross_entropy(prob_sample, label_sample)
                    train_loss.append(loss_train.item())
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                else:
                    test_size += person_input.shape[0]
                    # loss = F.cross_entropy(prob, train_label)
                    cum_test += calculate_acc(prob, train_label)
                    cum_ACC += calculate_precision(prob, train_label)
                    cum_Mrr.append(calculate_mrr(prob, train_label))
        if config["no_clip"]:
            pass

        if config["mean_clip"]:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip"])  # 裁剪
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.normal(0,
                                         (2 * 0.01 * config["clip"]) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5 * epsilon,
                                         size=param.grad.shape).to(device)
                    param.grad += noise

        elif config["median_clip"]:
            gradient_norms = []
            for param in model.parameters():
                if param.requires_grad:
                    gradient_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
                    gradient_norms.append(gradient_norm)
            clip = np.median(gradient_norms)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)  # 裁剪

            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.normal(0, (2 * 0.01 * clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5 * epsilon,
                                         size=param.grad.shape).to(device)
                    param.grad += noise

        elif config["Zhang_clip"]:
            clip = config["clip"]*exp(-0.1*config["server_round"])
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)  # 裁剪

            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.normal(0, (2 * 0.01 * clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5 * epsilon,
                                         size=param.grad.shape).to(device)
                    param.grad += noise

        elif not pre_grad_l2norms:
            count = 0
            for param in model.parameters():
                if param.requires_grad:
                    pre_grad_l2norms[count] = param.grad
                    count += 1
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.normal(0,
                                         (2 * 0.01 * config["clip"]) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5 * epsilon,
                                         size=param.grad.shape).to(device)
                    param.grad += noise

        else:
            count = 0
            for param in model.parameters():
                if param.requires_grad:
                    diff = pre_grad_l2norms[count].flatten() - param.grad.flatten()
                    R = torch.norm(diff, p=2)
                    pre_grad_l2norms[count] = param.grad
                    count += 1
                    clip = config["clip"] * exp(-t / (R.item()))
                    nn.utils.clip_grad_norm_(param.grad, max_norm=clip)

                    noise = torch.normal(0, (2 * 0.01 * clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5 * epsilon,
                                         size=param.grad.shape).to(device)
                    param.grad += noise

        scheduler.step()
        cum_test = np.array(cum_test) / test_size
        cum_ACC = np.array(cum_ACC) / test_size
        test_Acc.append(cum_ACC)
        test_acc.append(cum_test)
        loss_t.append(sum(train_loss) / len(train_loss))
        test_Mrr.append(sum(cum_Mrr) / len(cum_Mrr))
        print(
            f"\t Epoch {t}. Avg Loss: {sum(train_loss) / len(train_loss):.3f}"
        )

        # Update max_test_acc and min_loss_test
        max_test_acc = max(max_test_acc, cum_test, key=lambda x: sum(x))
        max_test_Acc = max(max_test_Acc, cum_ACC, key=lambda x: sum(x))
        max_Mrr = max(max_Mrr, np.max(test_Mrr))
        min_loss_test = min(min_loss_test, np.min(loss_t))

    return max_test_acc, max_test_Acc, max_Mrr, min_loss_test


class DP_Client(fl.client.NumPyClient):
    def __init__(self, model, trainloader, mat2s, cid):
        super().__init__()
        self.mat2s = mat2s
        self.cid = cid
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        self.model = model
        self.trainloader = trainloader

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        Config = {"server_round": config["server_round"], "epsilon": config["epsilon"],
                  "num_round": config["num_round"],
                  "enable_avg": config["enable_avg"],
                  "enable_seq": config["enable_seq"],
                  "enable_ser": config["enable_ser"],
                  "mean_clip": config["mean_clip"],
                  "clip": config["clip"],
                  "median_clip": config["median_clip"],
                  "no_clip": config["no_clip"],
                  "Zhang_clip": config["Zhang_clip"]}

        test_acc, test_Acc, Mrr, loss = train(model=self.model, trainloader=self.trainloader,
                                              mat2s=self.mat2s,
                                              cid=self.cid, optimizer=self.optimizer, config=Config)

        return (
            self.get_parameters(config),
            len(self.trainloader),
            {"test_acc": test_acc, "test_Acc": test_Acc, "Mrr": Mrr, "loss": loss, },
        )
