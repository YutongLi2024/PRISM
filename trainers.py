# -*- coding: utf-8 -*-
import numpy as np
import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric, generate_padded_sequences_tensor
from modules import wasserstein_distance, kl_distance, wasserstein_distance_matmul
from STOSA import STOSA
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        recall_dict_list = []
        ndcg_dict_list = []
        
        for k in [10, 20]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        
        # post_fix = {
        #     "Epoch": epoch,
        #     "Recall@10": '{:.8f}'.format(recall[0]), "NDCG@10": '{:.8f}'.format(ndcg[0]),
        #     "Recall@20": '{:.8f}'.format(recall[1]), "NDCG@20": '{:.8f}'.format(ndcg[1]),
        # }
        post_fix = {
            "Epoch": epoch,
            "Recall@10": '{:.4f}'.format(recall[0]), 
            "Recall@20": '{:.4f}'.format(recall[1]),
            "NDCG@10": '{:.4f}'.format(ndcg[0]), 
            "NDCG@20": '{:.4f}'.format(ndcg[1]),
        }
        
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        
        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix), [recall_dict_list, ndcg_dict_list]


    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0

            # for i, batch in rec_data_iter:
            for batch in tqdm(rec_data_iter):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                # sequence_output, _ = self.model.finetune(input_ids)
                sequence_output, _, total_interaction_loss = self.model.finetune(input_ids)
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                loss += total_interaction_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output, _,_ = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in tqdm(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    # print(input_ids)
                    recommend_output,_ ,_= self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)


class STOSATrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(STOSATrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):  
        # print("bpr_optimization")
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)
        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict_full(self, seq_mean_out, seq_cov_out):  
        # print("dist_predict_full")
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1

        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0

            # for batch in rec_data_iter:
            for batch in tqdm(rec_data_iter, desc="Training Progress"):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                # bpr optimization
                sequence_mean_output, sequence_cov_output, _, _, interaction_losse = self.model.finetune(input_ids, user_ids)
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)

                # loss = loss + pvn_loss
                loss = loss + pvn_loss + interaction_losse
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    # for i, batch in rec_data_iter:
                    i = 0
                    for batch in tqdm(rec_data_iter):
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _, _, _= self.model.finetune(input_ids, user_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)



class InDiRecTrainer(Trainer):
    def __init__(self, model, train_dataloader,cluster_dataloader,eval_dataloader, test_dataloader, device, args):
        super(InDiRecTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)
        self.cluster_dataloader = cluster_dataloader
        self.device = device

    def iteration(self, epoch, dataloader,cluster_dataloader=None, train=True):
        if train:
            print("Preparing Clustering:")
            self.model.eval()
            kmeans_training_data = []
            sequence_ids = []
            sequences_list = []
            sequence_counter = 0  

            rec_t_data_iter = tqdm(enumerate(self.cluster_dataloader), total=len(self.cluster_dataloader))
            for i, (rec_batch) in rec_t_data_iter: 
                
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_id, subsequence, _, _, _ = rec_batch 
                batch_size = subsequence.size(0)

                sequence_output_a = self.seq_model(subsequence) 
                sequence_output_b=sequence_output_a[:,-1,:]  
                kmeans_training_data.append(sequence_output_b.detach().cpu().numpy()) 
                sequences_list.append(subsequence.detach().cpu().numpy())

                batch_sequence_ids = np.arange(sequence_counter, sequence_counter + batch_size)
                sequence_counter += batch_size
                sequence_ids.extend(batch_sequence_ids)

            sequences_list = np.concatenate(sequences_list, axis=0)
            sequence_ids = np.array(sequence_ids)

            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
            kmeans_training_data_t = [kmeans_training_data]

            sequences_list_t = [sequences_list]
            sequence_ids_t = [sequence_ids]

            for i, clusters in enumerate(self.clusters_t):
                for j, cluster in enumerate(clusters):
                    cluster.train(kmeans_training_data_t[i], sequence_ids_t[i])
                    self.clusters_t[i][j] = cluster

            self.sequences_list = sequences_list_t[0]

            # clean memory
            del kmeans_training_data
            del kmeans_training_data_t

            del sequences_list
            del sequences_list_t
            del sequence_ids
            del sequence_ids_t
            gc.collect()
            

            self.model.train()
            avg_loss = 0.0

            # print(f"rec dataset length: {len(dataloader)}")
            rec_t_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # minibatch
            for i, (rec_batch) in rec_t_data_iter:

                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_seq, input_seq_len, target_pos, target = rec_batch
                
                same_cl_seqs =  self.get_sequences_in_same_cluster(input_seq)
                same_cl_seqs = np.array(same_cl_seqs)
                same_cl_seqs = torch.tensor(same_cl_seqs, dtype=torch.long).to(self.device)

                x0_gt = self.model.calculate_x0(target) # B x D
                
                s = self.model.calculate_s(same_cl_seqs, self.args.p)

                t = torch.randint(0, self.args.timesteps, (self.args.batch_size, ), device=self.device).long()
                diff_loss, predicted_x_0 = self.calculate_diff_loss(self.model, x0_gt, s, t)
                
                # ablation
                if self.args.diff_weight == 0:
                    sequences_tensor = generate_padded_sequences_tensor(self.args.batch_size, self.args.max_seq_length ,self.args.item_size, input_seq_len)
                    sequences_tensor =sequences_tensor.to(self.device)
                    x0_hat_out = self.seq_model(sequences_tensor)
                    x0_hat_out = x0_hat_out[:,-1,:]
                
                else:
                    s_hat = self.seq_model(same_cl_seqs)
                    x0_hat_out = self.model.sample_from_reverse_process(s_hat[:,-1,:]) # BxD

                x0_out = self.seq_model(input_seq) # BxLxD

                cl_loss = self.calculate_cl_loss(x0_hat_out, x0_out[:,-1,:], target_pos[:, -1])

                logits = self.predict_full(x0_out[:, -1, :])  #  Bx|I| 
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])

                multi_task_loss = self.args.rec_weight*rec_loss + self.args.diff_weight*diff_loss + self.args.cl_weight*cl_loss
                

                self.optimizer.zero_grad()
                multi_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                avg_loss += rec_loss.item()
                
            log_record = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(avg_loss / len(rec_t_data_iter)),
            }

            if (epoch+1) % self.args.print_log_freq == 0:
                print(str(log_record))

            with open(self.args.log_file, "a") as f:
                f.write(str(log_record) + "\n")

        else: # Val or Test
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader)) 
            self.model.eval()
            pred_list, answer_list = None, None


            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids,input_seq_len, target_pos, answers = batch
                rec_output = self.seq_model(input_ids)
                rec_output = rec_output[:,-1,:]


                rating_pred = self.predict_full(rec_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()

                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -20)[:, -20:] 
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind] 
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1] 
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()

                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0) 

            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)


            return metrices_list, log_info
                    
    
    def calculate_diff_loss(self, model, x_start, s, t, noise=None, loss_type="l2"):

        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = model.forward_process(x_start=x_start, t=t, noise=noise)

        predicted_x = model(x_noisy, s, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x 
    
    def calculate_cl_loss(self, x0_hat_rep, x0_rep, target):
        batch_size = x0_rep.shape[0]
        sem_nce_logits, sem_nce_labels = self.info_nce(x0_hat_rep,x0_rep,self.args.temperature, batch_size, self.sim, target)
        cl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(intent_id)
        negative_samples = sim
        negative_samples[mask==0]=float("-inf")

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def mask_correlated_samples(self, label):
        label=label.view(1,-1)
        label=label.expand((2,label.shape[-1])).reshape(1,-1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask==0
