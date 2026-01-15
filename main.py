# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import SASRecDataset, D,D_random, get_seqs_and_matrixes, DatasetForInDiRec
from trainers import SASRecTrainer, STOSATrainer, InDiRecTrainer
from STOSA import STOSA
from SASRec import SASRecModel
from InDiRec import InDiRec
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
import time
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  
data_name = 'Beauty'

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=f'./data/{data_name}/', type=str)
    parser.add_argument('--output_dir', default=f'outputs/{data_name}', type=str)
    parser.add_argument('--data_name', default=f'reviews_{data_name}', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--patience', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='STOSA', type=str, help="SASRec/STOSA/InDiRec")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size of transformer model") #64
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # multimodal args
    parser.add_argument('--image_emb_path', default=f'data/{data_name}/image_features_{data_name}.pt', type=str)
    parser.add_argument('--text_emb_path', default=f'data/{data_name}/text_features_{data_name}.pt', type=str)
    parser.add_argument("--pretrain_emb_dim", type=int, default=512, help="pretrain_emb_dim of clip model")
    # parser.add_argument("--pretrain_emb_dim", type=int, default=768, help="pretrain_emb_dim of clip model")
    parser.add_argument("--prediction", type=bool, default=False, help="activate prediction mode")
    parser.add_argument("--lambda_uni", type=int, default=0.1, help="the weigth of uniqueness_losses")
    parser.add_argument("--lambda_syn", type=int, default=0.1, help="the weigth of synergy_loss")
    parser.add_argument("--lambda_red", type=int, default=0.1, help="the weigth of redundancy_loss")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--device", type=str, default="cuda:0", help="train device")

    parser.add_argument("--model_idx", default=1, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--rec_weight", type=float, default=1,  help="weight of rating prediction")
    parser.add_argument("--diff_weight", type=float, default=1, help="weight of intent-aware diffusion task")
    parser.add_argument("--cl_weight", type=float, default=0.4, help="weight of contrastive learning task")
    parser.add_argument('--w', type=float, default=2.0, help='control the strength of intent-guided signal s')
    parser.add_argument("--intent_num",default=256,type=int, help="the multi intent nums.")
    parser.add_argument("--sim",default='dot',type=str, help="the calculate ways of the similarity.")

    # diffusion
    parser.add_argument('--timesteps', type=int, default=200, help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta start of diffusion')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='type of diffuser.')
    parser.add_argument('--beta_sche', nargs='?', default='exp', help='beta schedule')
    parser.add_argument('--without_segment', action="store_true", help='dropout ')
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1


    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.lambda_uni}-{args.lambda_syn}-{args.lambda_red}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)

    if args.model_name == 'STOSA':
        model = STOSA(args=args)
        test_real_memory_usage(model, args, 'STOSA')
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = STOSATrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'SASRec':
        model = SASRecModel(args=args)
        test_real_memory_usage(model, args, 'SASRec')
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = SASRecTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)
    else:
        model = InDiRec(args=args)
        test_real_memory_usage(model, args, 'InDiRec')
        if args.without_segment:
            args.segmented_file = args.data_dir + args.data_name + "_random.txt"
            D_random(args.data_file,args.segmented_file,args.max_seq_length)

        else:
            args.segmented_file = args.data_dir + args.data_name + "_s.txt"
        
        if not os.path.exists(args.segmented_file):
            D(args.data_file,args.segmented_file,args.max_seq_length)

        _,train_seq = get_seqs_and_matrixes("training", args.segmented_file)
        _,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_seqs_and_matrixes("rating", args.data_file)

    
        cluster_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=True)

        training_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
        training_sampler = RandomSampler(training_dataset)
        training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=args.batch_size, drop_last=True)

        eval_dataset = DatasetForInDiRec(args, user_seq, data_type="valid")
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

        testing_dataset = DatasetForInDiRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = InDiRecTrainer(model, training_dataloader, cluster_dataloader, eval_dataloader, testing_dataloader, device, args)

            
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        if args.model_name == 'STOSA':
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        else:
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            start_time_epoch_valid = time.time()
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)

            end_time_epoch_valid = time.time()
            epoch_duration = end_time_epoch_valid - start_time_epoch_valid
            x_epoch_duration = format(epoch_duration, ".2f")
            print(f"Epoch {epoch} valid in {epoch_duration:.2f} seconds.")

            with open(args.log_file, 'a') as f:
                f.write(f"Epoch {epoch} duration: {epoch_duration:.2f} seconds\n")
                f.write(x_epoch_duration + '\n')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')

        # Load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix

        # Start timing the testing phase
        start_time_test = time.time()
        scores, result_info, _ = trainer.test('best', full_sort=True)
        end_time_test = time.time()

        # Calculate and log the prediction time
        prediction_duration = end_time_test - start_time_test
        print(f"Prediction time: {prediction_duration:.2f} seconds.")
        with open(args.log_file, 'a') as f:
            f.write(f"Prediction time: {prediction_duration:.2f} seconds\n")

    # Log total training time
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
        f.write(f"Total training time: {int(minutes):02d}:{int(seconds):02d}" + '\n')

    print(f"Total training time: {int(minutes):02d}:{int(seconds):02d}")

def test_real_memory_usage(model, args, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    batch_size = args.batch_size
    seq_len = args.max_seq_length
    input_ids = torch.randint(1, args.item_size, (batch_size, seq_len)).to(device)
    user_ids = torch.randint(0, args.num_users, (batch_size,)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        outputs = model.finetune(input_ids, user_ids)
    except TypeError:
        outputs = model.finetune(input_ids)

    if isinstance(outputs, tuple):
        loss = outputs[-1] if torch.is_tensor(outputs[-1]) else outputs[0]
    else:
        loss = outputs
    loss = loss.mean() if loss.dim() > 0 else loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"memory:  {mem_peak:.2f} GB\n")

    torch.cuda.empty_cache()



main()



