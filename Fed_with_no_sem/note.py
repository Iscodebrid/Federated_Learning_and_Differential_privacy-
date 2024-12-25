'''
@Project ：pythonProject1 
@File    ：note.py
@IDE     ：PyCharm 
@Author  ：Yangjie
@Date    ：2024/4/25 9:52 
@清华源   ：-i https://pypi.tuna.tsinghua.edu.cn/simple
'''
# def insert_evaluation_result_to_csv(ser_round, num_clients, v_acc, test_acc):
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([ser_round, num_clients, v_acc, test_acc])


# def get_evaluate_fn():
#     """Return an evaluation function for server-side evaluation."""
#
#     # The `evaluate` function will be called after every round
#     def evaluate(server_round, parameters: fl.common.NDArrays, config):
#         if server_round == 0:
#             return None
#         # Update model with the latest parameters
#         print(f"{server_round} ----- evaluate")
#         testdata = load_data(2)
#         eval_loader = torch.utils.data.DataLoader(testdata, args.batch, shuffle=False)
#
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)
#
#         model.to(DEVICE)
#
#         for t in range(args.e_epochs):
#             # valid_size = 0.0
#             # acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
#             cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]
#             # loss, accuracy = test(model=model, data_loader=eval_loader, mat2s=mat2s)
#             for step, item in enumerate(eval_loader):
#
#                 # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
#                 person_input, person_m1, person_m2t, person_label, person_traj_len = item
#
#                 locations = person_input[:, :, 1]
#
#                 # Flatten locations to 1D tensor
#                 locations_flat = locations.view(-1)
#
#                 # Ensure data type is integer and non-negative
#                 locations_flat = locations_flat.to(torch.int32).clamp(min=0)
#
#                 # Transfer tensor to CPU and convert to NumPy array
#                 locations_flat_cpu = locations_flat.cpu().numpy()
#
#                 # Use Counter to count occurrences
#                 counts_dict = Counter(locations_flat_cpu)
#
#                 # Convert counts to a tensor
#                 counts_tensor = torch.tensor(list(counts_dict.values()), dtype=torch.int32).to(device)
#
#                 padding_length = 100 - len(counts_tensor)
#
#                 # If padding is needed, create a tensor of zeros and concatenate it with counts_tensor
#                 if padding_length > 0:
#                     padding = torch.zeros(padding_length, dtype=torch.int32, device=device)
#                     counts_tensor = torch.cat((counts_tensor, padding), dim=0)
#                 # first, try batch_size = 1 and mini_batch = 1
#                 counts_mask = torch.zeros_like(counts_tensor)
#                 input_mask = torch.zeros((args.batch, max_len, 3), dtype=torch.long).to(device)
#                 m1_mask = torch.zeros((args.batch, max_len, max_len, 2), dtype=torch.float32).to(device)
#                 for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
#                     # if mask_len != person_traj_len[0]:
#                     #     continue
#                     input_mask[:, :mask_len] = 1.
#                     m1_mask[:, :mask_len, :mask_len] = 1.
#
#                     train_input = person_input * input_mask
#                     train_m1 = person_m1 * m1_mask
#                     train_m2t = person_m2t[:, mask_len - 1]
#                     train_label = person_label[:, mask_len - 1]  # (N)
#                     train_len = torch.zeros(size=(args.batch,), dtype=torch.long).to(device) + mask_len
#
#                     counts_mask[:len(counts_tensor)] = 1
#                     counts = counts_tensor * counts_mask
#
#                     prob = model(train_input, train_m1, mat2s, train_m2t, train_len, counts)  # (N, L)
#
#                     if mask_len <= person_traj_len[0] - 2:  # only training
#                         continue
#
#                     elif mask_len == person_traj_len[0] - 1:  # only validation
#                         # valid_size += person_input[0]
#                         acc_valid = calculate_acc(prob, train_label)
#                         cum_valid += calculate_acc(prob, train_label)
#
#                     elif mask_len == person_traj_len[0]:  # only test
#                         acc_test = calculate_acc(prob, train_label)
#                         cum_test += calculate_acc(prob, train_label)
#             print(f"cum_valid{cum_valid / len(cum_valid)}")
#             # insert_evaluation_result_to_csv(server_round, args.NUM_CLIENTS, acc_valid, acc_test)
#
#         return {"there is no Loss": 0.0}, {"accuracy": None}
#
#     return evaluate


# def fit_weighted_average(metrics):
#     """Aggregation function for (federated) fit metrics."""
#     # Multiply accuracy of each client by number of examples used
#
#     losses = [m["loss"] for num_examples, m in metrics]
#     losses = np.mean(losses)
#
#     acc = [m["valid_acc"] for num_examples, m in metrics]
#     acc = np.mean(acc, axis=0)
#     # pdb.set_trace()
#     # 写入日志文件
#     write_to_log(losses, acc)
#     print(f"this round avg_loss: {losses}, this round avg_acc: {acc}")
#     # Aggregate and return custom metric (weighted average)
#     return {"train_loss": losses, "valid_acc": acc}

# def get_evaluate_fn():
#     """Return an evaluation function for server-side evaluation."""
#
#     # The `evaluate` function will be called after every round
#     def evaluate(server_round, parameters: fl.common.NDArrays, config):
#         if server_round == 0:
#             return None
#         # Update model with the latest parameters
#         print(f"{server_round} ----- evaluate")
#         testdata, model, mat2s = load_data(30, sid='eval')
#         eval_loader = torch.utils.data.DataLoader(testdata, args.batch, shuffle=False)
#
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)
#
#         model.to(DEVICE)
#
#         # Evaluate the model on the test dataset
#         with torch.no_grad():
#             cum_valid = []
#             Eval_Loss = []
#             for step, item in enumerate(eval_loader):
#                 # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
#                 person_input, person_m1, person_m2t, person_label, person_traj_len = item
#
#                 input_mask = torch.zeros((args.batch, max_len, 3), dtype=torch.long).to(args.DEVICE)
#                 m1_mask = torch.zeros((args.batch, max_len, max_len, 2), dtype=torch.float32).to(args.DEVICE)
#                 for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
#                     input_mask[:, :mask_len] = 1.
#                     m1_mask[:, :mask_len, :mask_len] = 1.
#
#                     eval_input = person_input * input_mask
#                     eval_m1 = person_m1 * m1_mask
#                     eval_m2t = person_m2t[:, mask_len - 1]
#                     eval_label = person_label[:, mask_len - 1]  # (N)
#                     eval_len = torch.zeros(size=(args.batch,), dtype=torch.long).to(args.DEVICE) + mask_len
#
#                     prob = model(eval_input, eval_m1, mat2s, eval_m2t, eval_len)  # (N, L)
#                     loss_eval = F.cross_entropy(prob, eval_label)
#                     Eval_Loss.append(loss_eval.item())
#                     cum_valid.append(calculate_acc(prob, eval_label))
#
#         avg_acc_valid = np.mean(cum_valid, axis=0)
#         avg_loss = sum(Eval_Loss) / len(Eval_Loss)
#         write_to_log(acc_valid=avg_acc_valid, loss=avg_loss)
#
#         print(f"Average validation accuracy: {avg_acc_valid}, Average loss: {avg_loss}")
#         return {"there is no Loss": 0.0}, {"accuracy": None}
#
#     return evaluate



# def get_evaluate_fn():
#     """Return an evaluation function for server-side evaluation."""
#
#     def evaluate(server_round, parameters: fl.common.NDArrays, config):
#         if server_round == 0:
#             return None
#
#         print(f"{server_round} ----- evaluate")
#         for cid in range(args.NUM_CLIENTS):
#
#             testdata, model, mat2s = load_data(cid=cid)
#
#             eval_loader = torch.utils.data.DataLoader(testdata, args.batch, shuffle=False)
#
#             params_dict = zip(model.state_dict().keys(), parameters)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             model.load_state_dict(state_dict, strict=True)
#
#             model.to(DEVICE)
#
#             # Initialize variables to store cumulative accuracy
#             cum_test = 0.0
#             test_size = 0
#             loss_valid = []
#             # Evaluate the model on the test dataset
#             with torch.no_grad():
#
#                 for step, item in enumerate(eval_loader):
#                     # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
#                     person_input, person_m1, person_m2t, person_label, person_traj_len = item
#
#                     input_mask = torch.zeros((args.batch, max_len, 3), dtype=torch.long).to(args.DEVICE)
#                     m1_mask = torch.zeros((args.batch, max_len, max_len, 2), dtype=torch.float32).to(args.DEVICE)
#                     for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
#                         input_mask[:, :mask_len] = 1.
#                         m1_mask[:, :mask_len, :mask_len] = 1.
#
#                         train_input = person_input * input_mask
#                         train_m1 = person_m1 * m1_mask
#                         train_m2t = person_m2t[:, mask_len - 1]
#                         train_label = person_label[:, mask_len - 1]  # (N)
#                         train_len = torch.zeros(size=(args.batch,), dtype=torch.long).to(args.DEVICE) + mask_len
#
#                         prob = model(train_input, train_m1, mat2s, train_m2t, train_len)  # (N, L)
#
#                         if mask_len <= person_traj_len[0] - 5:
#                             continue
#
#                         else:  # only test
#                             loss_valid.append(F.cross_entropy(prob, train_label).item())
#                             test_size += person_input.shape[0]
#                             cum_test += calculate_acc(prob, train_label)
#
#             avg_acc_test = cum_test / test_size
#
#             print(f"Average test accuracy: {avg_acc_test}, Avg loss:{sum(loss_valid) / len(loss_valid)}")
#
#         return {"there is no Loss": 0.0}, {"accuracy": None}
#
#     return evaluate


#     # user 300-350 to eval global model
#     trajs, mat1, mat2t, labels, lens = \
#         (trajs[cid:cid + 10], mat1[cid:cid + 10], mat2t[cid:cid + 10],
#          labels[cid:cid + 10], lens[cid:cid + 10])
#
# else:
#     # this means each client have 30 len data


# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#             self,
#             server_round: int,
#             results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
#             failures: List[
#                 Union[
#                     Tuple[client_proxy.ClientProxy, fl.common.FitRes],
#                     BaseException,
#                 ]
#             ],
#     ) -> Optional[fl.common.NDArrays]:
#         weights = super().aggregate_fit(server_round, results, failures)
#
#         # 预留, 对服务器梯度的操作
#         return weights


# def load_data(cid, sid):
#     [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = data
#     mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(DEVICE), \
#         torch.FloatTensor(mat2t), torch.LongTensor(lens)
#
#     if sid == 'eval':
#         # user 300-350 to eval global model
#         trajs, mat1, mat2t, labels, lens = \
#             (trajs[cid:cid + 10], mat1[cid:cid + 10], mat2t[cid:cid + 10],
#              labels[cid:cid + 10], lens[cid:cid + 10])
#
#     if sid == 'train':
#         user_data_size = 30
#         start_index = cid * user_data_size
#         end_index = start_index + user_data_size
#         trajs, mat1, mat2t, labels, lens = \
#             (trajs[start_index:end_index], mat1[start_index:end_index], mat2t[start_index:end_index],
#              labels[start_index:end_index], lens[start_index:end_index])
#
#     ex = (mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min())
#     model = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, embed_dim=50)
#     dataset = CustomDataset(traj=trajs, m1=mat1, v=mat2t, label=labels - 1, length=lens)
#
#     return dataset, model, mat2s, ex


# def get_evaluate_fn():
#     """Return an evaluation function for server-side evaluation."""
#
#     def evaluate(server_round, parameters: fl.common.NDArrays, config):
#         if server_round == 0:
#             return None
#
#         print(f"{server_round} ----- evaluate")
#         # for cid in range(args.NUM_CLIENTS):
#
#         testdata, model, mat2s, ex = load_data(cid=400, sid='eval')
#
#         eval_loader = torch.utils.data.DataLoader(testdata, args.batch, shuffle=False)
#
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)
#
#         model.to(DEVICE)
#
#         # Initialize variables to store cumulative accuracy
#         cum_test = 0.0
#         test_size = 0
#         loss_valid = []
#         # Evaluate the model on the test dataset
#         with torch.no_grad():
#
#             for step, item in enumerate(eval_loader):
#                 # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
#                 person_input, person_m1, person_m2t, person_label, person_traj_len = item
#
#                 input_mask = torch.zeros((args.batch, max_len, 3), dtype=torch.long).to(args.DEVICE)
#                 m1_mask = torch.zeros((args.batch, max_len, max_len, 2), dtype=torch.float32).to(args.DEVICE)
#                 for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
#                     input_mask[:, :mask_len] = 1.
#                     m1_mask[:, :mask_len, :mask_len] = 1.
#
#                     train_input = person_input * input_mask
#                     train_m1 = person_m1 * m1_mask
#                     train_m2t = person_m2t[:, mask_len - 1]
#                     train_label = person_label[:, mask_len - 1]  # (N)
#                     train_len = torch.zeros(size=(args.batch,), dtype=torch.long).to(args.DEVICE) + mask_len
#
#                     prob = model(train_input, train_m1, mat2s, train_m2t, train_len, ex)  # (N, L)
#
#                     # if mask_len <= person_traj_len[0] - 5:
#                     #     continue
#                     #
#                     # else:  # only test
#                     loss_valid.append(F.cross_entropy(prob, train_label).item())
#                     test_size += person_input.shape[0]
#                     cum_test += calculate_acc(prob, train_label)
#
#         avg_acc_test = cum_test / test_size
#
#         print(f"Average test accuracy: {avg_acc_test}, Avg loss:{sum(loss_valid) / len(loss_valid)}")
#
#         return {"there is no Loss": 0.0}, {"accuracy": None}
#
#     return evaluate
# if config["mean_clip"]:
#     # clipping -- add Noise
#     gradient_norms = []
#     for param in model.parameters():
#         if param.requires_grad:
#             gradient_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#             gradient_norms.append(gradient_norm)

    # for param in model.parameters():
    #     if param.requires_grad:
    #         noise = torch.normal(0, clip * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
    #                              size=param.grad.shape).to(device)
    #         param.grad += noise
#
# else:
#
#     if not pre_grad_l2norms:
#         i = 0
#         for param in model.parameters():
#             if param.requires_grad:
#                 l2_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#                 pre_grad_l2norms[i] = l2_norm
#                 i += 1
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # 裁剪
#
#         # for param in model.parameters():
#         #     if param.requires_grad:
#         #         noise = torch.normal(0, 10 * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#         #                              size=param.grad.shape).to(device)
#         #         param.grad += noise
#     else:
#         for param in model.parameters():
#             i = 0
#             if param.requires_grad:
#                 l2_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#                 clip = pre_grad_l2norms[i] - l2_norm
#                 pre_grad_l2norms[i] = l2_norm
#                 clip = np.median(param.grad) * exp(-t / clip)
#                 print(clip)
#                 nn.utils.clip_grad_norm_(param.grad, max_norm=np.median(param.grad) * exp(-t / clip))  # 裁剪
#
#                 noise = torch.normal(0, clip * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#                                      size=param.grad.shape).to(device)
#                 param.grad += noise
# elif not pre_grad_l2norms:
# gradient_norms = []
# for param in model.parameters():
#     if param.requires_grad:
#         gradient_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#         gradient_norms.append(gradient_norm)
# pre_grad_l2norms["clip"] = np.median(gradient_norms)
#
# for param in model.parameters():
#     if param.requires_grad:
#         noise = torch.normal(0, (2 * 0.01 * pre_grad_l2norms["clip"]) * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#                              size=param.grad.shape).to(device)
#         param.grad += noise
# else:
#     gradient_norms = []
# for param in model.parameters():
#     if param.requires_grad:
#         gradient_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#         gradient_norms.append(gradient_norm)
# R = abs(pre_grad_l2norms["clip"] - np.median(gradient_norms))
# pre_grad_l2norms["clip"] = np.median(gradient_norms)
# clip = config["clip"] * exp(-t / R)
#
# nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
#
# for param in model.parameters():
#     if param.requires_grad:
#         noise = torch.normal(0, (2 * 0.01 * clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#                              size=param.grad.shape).to(device)
#         param.grad += noise


# def train(model, trainloader, mat2s, cid, optimizer, config):
#     model.to(device)
#     num_neg = 10  # n neg
#     print(f"  ---client_{cid}--start train---  ")
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=1)
#     loss_test = []
#     test_acc = []
#     pre_grad_l2norms = {}
#     if config["enable_avg"]:
#         epsilon = config["epsilon"] / config["num_round"]
#         epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
#
#     elif config["server_round"] == 1:
#         epsilon = 1.5
#         epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
#     else:
#         epsilon = 2 * sqrt((1.5 / 2) ** 2 + (0.7 + 0.22 * (config["server_round"] - 2)) ** 2)
#         epochs = moments_calcu(max_lmbd=32, q=0.01, sigma=0.9, epsilon=epsilon)
#
#     for t in range(1, epochs+1):
#         train_loss = []
#         cum_test = [0, 0, 0, 0]
#         test_size = 0
#         test_loss = []
#         for step, item in enumerate(trainloader):
#
#             # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
#             person_input, person_m1, person_m2t, person_label, person_traj_len = item
#
#             # first, try batch_size = 1 and mini_batch = 1
#             # pdb.set_trace()
#             input_mask = torch.zeros((PARAMS["batch_size"], max_len, 3), dtype=torch.long).to(device)
#             m1_mask = torch.zeros((PARAMS["batch_size"], max_len, max_len, 2), dtype=torch.float32).to(device)
#
#             for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
#
#                 input_mask[:, :mask_len] = 1.
#                 m1_mask[:, :mask_len, :mask_len] = 1.
#
#                 train_input = person_input * input_mask
#                 train_m1 = person_m1 * m1_mask
#                 train_m2t = person_m2t[:, mask_len - 1]
#                 train_label = person_label[:, mask_len - 1]  # (N)
#                 train_len = torch.zeros(size=(PARAMS["batch_size"],), dtype=torch.long).to(device) + mask_len
#
#                 prob = model(train_input, train_m1, mat2s, train_m2t, train_len)  # (N, L)
#
#                 if mask_len <= person_traj_len[0] * 2 / 3:
#                     prob_sample, label_sample = sampling_prob(prob, train_label, num_neg)
#                     loss_train = F.cross_entropy(prob_sample, label_sample)
#                     train_loss.append(loss_train.item())
#                     optimizer.zero_grad()
#                     loss_train.backward()
#                     optimizer.step()
#                 else:
#                     test_size += person_input.shape[0]
#                     loss = F.cross_entropy(prob, train_label)
#                     test_loss.append(loss.item())
#                     cum_test += calculate_acc(prob, train_label)
#
#         if config["mean_clip"]:
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip"])  # 裁剪
#             for param in model.parameters():
#                 if param.requires_grad:
#                     noise = torch.normal(0, (2 * 0.01 * config["clip"]) * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#                                          size=param.grad.shape).to(device)
#                     param.grad += noise
#
#         elif config["median_clip"]:
#             gradient_norms = []
#             for param in model.parameters():
#                 if param.requires_grad:
#                     gradient_norm = torch.norm(param.grad, p=2).item()  # 计算梯度的L2范数
#                     gradient_norms.append(gradient_norm)
#             clip = np.median(gradient_norms)
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)  # 裁剪
#
#             for param in model.parameters():
#                 if param.requires_grad:
#                     noise = torch.normal(0, (2 * 0.01 * clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon,
#                                          size=param.grad.shape).to(device)
#                     param.grad += noise
#         elif not pre_grad_l2norms:
#             count = 0
#             for param in model.parameters():
#                 if param.requires_grad:
#                     pre_grad_l2norms[count] = param.grad
#                     count += 1
#             for param in model.parameters():
#                 if param.requires_grad:
#                     noise = torch.normal(0, (2*0.01*config["clip"]) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5*epsilon,
#                                          size=param.grad.shape).to(device)
#                     param.grad += noise
#         else:
#             count = 0
#             for param in model.parameters():
#                 if param.requires_grad:
#                     diff = pre_grad_l2norms[count].flatten()-param.grad.flatten()
#                     R = torch.norm(diff, p=2)
#                     pre_grad_l2norms[count] = param.grad
#                     count += 1
#                     clip = config["clip"]*exp(-t/(R.item()))
#                     nn.utils.clip_grad_norm_(param.grad, max_norm=clip)
#
#                     noise = torch.normal(0, (2*0.01*clip) * np.sqrt(2 * np.log(1.25 / 1e-5)) / 5*epsilon,
#                                          size=param.grad.shape).to(device)
#                     param.grad += noise
#
#         scheduler.step()
#         cum_test = np.array(cum_test) / test_size
#         test_acc.append(cum_test)
#         loss_test.append(np.mean(test_loss))
#         print(
#             f"\t Epoch {t}. Avg Loss: {sum(train_loss) / len(train_loss):.3f}"
#         )
#     return sum(test_acc) / len(test_acc), sum(loss_test) / len(loss_test)  # we use this to eval the result



