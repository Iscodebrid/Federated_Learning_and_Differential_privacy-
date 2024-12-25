'''
@Project ：GETNext-master 
@File    ：dp_simulation.py
@IDE     ：PyCharm 
@Author  ：Yangjie
@Date    ：2024/3/11 13:31 
@清华源   ：-i https://pypi.tuna.tsinghua.edu.cn/simple
'''
import pdb
from datetime import datetime
import numpy as np
from typing import Dict
from flwr.common.typing import Scalar
import flwr as fl
import joblib
import torch
from flwr.client import Client
from torch.utils.data import Dataset
from dp_main import DP_Client, Model

# Define parameters.
from layers import hours
import argparse


""
"""
parameter group we defined
"""
parser = argparse.ArgumentParser(description='管理参数集合')
parser.add_argument('--NUM_CLIENTS', type=int, help='客户端数量')
parser.add_argument('--name', type=str, help='数据集名称')
parser.add_argument('--DEVICE', type=str, help='device')
parser.add_argument('--batch', type=int, help='batch size')
args = parser.parse_args()
args.DEVICE = "cuda:0"
args.NUM_CLIENTS = 10
args.name = "NYC"
args.batch = 1
DEVICE = args.DEVICE
""


class CustomDataset(Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(DEVICE)
        mats1 = self.mat1[index].to(DEVICE)
        vector = self.vec[index].to(DEVICE)
        label = self.label[index].to(DEVICE)
        length = self.length[index].to(DEVICE)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


def write_to_log(acc_test, Acc_test, Mrr, loss):
    """Write losses and acc to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("training_log.txt", "a") as f:
        f.write(f"{timestamp} - acc_test: {acc_test}, -Acc_test: {Acc_test}, -Mrr: {Mrr} loss: {loss}\n")


def fit_weighted_average(metrics):
    """Aggregation function for (federated) fit metrics."""

    losses = [m["loss"] for num_examples, m in metrics]
    losses = np.min(losses)

    Mrr = [m["Mrr"] for num_examples, m in metrics]
    Mrr = np.max(Mrr, axis=0)

    acc = [m["test_acc"] for num_examples, m in metrics]
    acc = np.max(acc, axis=0)

    Acc = [m["test_Acc"] for num_examples, m in metrics]
    Acc = np.max(Acc, axis=0)

    write_to_log(acc_test=acc, Acc_test=Acc, Mrr=Mrr, loss=losses)
    print(f"this round avg_loss: {losses}, this round Recall: {acc}, Acc:{Acc}, Mrr: {Mrr}")

    return {"train_loss": losses, "valid_acc": acc}


def client_fn(cid: str) -> Client:
    # 加载模型
    cid = int(cid)
    dataset, model, mat2s = load_data(cid=cid, sid='train')
    client_trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=False)
    return DP_Client(model, client_trainloader, mat2s, cid).to_client()


def load_data(cid, sid):
    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = data
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(DEVICE), \
        torch.FloatTensor(mat2t), torch.LongTensor(lens)

    if sid == 'train':
        user_data_size = 30
        start_index = cid * user_data_size
        end_index = start_index + user_data_size
        trajs, mat1, mat2t, labels, lens = \
            (trajs[start_index:end_index], mat1[start_index:end_index], mat2t[start_index:end_index],
             labels[start_index:end_index], lens[start_index:end_index])

    ex = (mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min())
    model = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, embed_dim=50, ex=ex)
    dataset = CustomDataset(traj=trajs, m1=mat1, v=mat2t, label=labels - 1, length=lens)

    return dataset, model, mat2s


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""

    config = {
        "server_round": server_round,
        "clip": 10,
        "epsilon": 20,
        "num_round": 10,
        "enable_avg": True,
        "enable_seq": False,
        "enable_ser": False,
        "median_clip": False,
        "mean_clip": False,
        "no_clip": True,
        "Zhang_clip": False,

    }
    return config


def main() -> None:
    # fl.common.logger.configure(identifier="Experiment for Fed point of interesting recommend",
    #                            filename="server_log.txt")

    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.NUM_CLIENTS,
        client_resources={"num_cpus": 12.0, "num_gpus": 1.0},
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )


if __name__ == "__main__":
    file = open('./data/' + args.name + '_data.pkl', 'rb')
    data = joblib.load(file)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.0,  # disable the evaluate function
        on_fit_config_fn=fit_config,
        # evaluate_fn=get_evaluate_fn(),
        fit_metrics_aggregation_fn=fit_weighted_average,
    )

    main()


