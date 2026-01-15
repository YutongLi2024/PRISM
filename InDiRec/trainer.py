import numpy as np
from tqdm import tqdm
import random
import gc
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import KMeans
from utils import recall_at_k, ndcg_k, generate_padded_sequences_tensor


class Trainer:
    def __init__(self, model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model
        self.seq_model = self.model.seq_model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=64,
            device=torch.device("cuda"),
        )
        self.clusters = [cluster]
        self.clusters_t=[self.clusters]

        self.train_dataloader = train_dataloader
        self.cluster_dataloader=cluster_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.seq_model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader,self.cluster_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HR@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HR@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    
    @torch.no_grad()
    def get_sequences_in_same_cluster(self, input_sequence):
        input_sequence = input_sequence.to(self.device)
        batch_size = input_sequence.size(0)
        sequence_output = self.seq_model(input_sequence)
        sequence_representation = sequence_output[:, -1, :]  

        sequence_representation_np = sequence_representation.cpu().numpy()
        cluster = self.clusters_t[0][0]  
        D, I = cluster.index.search(sequence_representation_np, 1)
        cluster_idx = int(I[0][0])
        sequence_ids_in_cluster = cluster.cluster2sequences[cluster_idx]
        sequences_in_cluster = [self.sequences_list[seq_id] for seq_id in sequence_ids_in_cluster]
        random_sequences = random.choices(sequences_in_cluster, k=batch_size)

        return random_sequences
    

class InDiRecTrainer(Trainer):
    def __init__(self, model, train_dataloader,cluster_dataloader,eval_dataloader, test_dataloader, device, args):
        super(InDiRecTrainer, self).__init__(model, train_dataloader, cluster_dataloader,eval_dataloader, test_dataloader, device, args)

    def iteration(self, epoch, dataloader,cluster_dataloader=None, train=True):
        if train:
            print("Preparing Clustering:")
            self.model.eval()
            kmeans_training_data = []
            sequence_ids = []
            sequences_list = []
            sequence_counter = 0  

            rec_t_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
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

