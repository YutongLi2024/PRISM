import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, DistSAEncoder
from modules import LayerNorm, DistSAEncoder
from copy import deepcopy
from PRISM import *



class STOSA(nn.Module):
    def __init__(self, args):
        super(STOSA, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_margins = nn.Embedding(args.num_users, 1)
        self.item_encoder = DistSAEncoder(args)
        self.interest_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # text
        self.text_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.text_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        # image
        self.image_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.image_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.fc_mean_image = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_cov_image = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_mean_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_cov_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)


        self.interaction_expert_layer = Interaction_Expert_Layer(
            args = args,
            hidden_size = args.hidden_size,
            expert_hidden_size = 128,
            num_experts_per_type=1)
        self.adaptive_fusion_layer = Adaptive_Fusion_Layer(args)

        self.apply(self.init_weights)

        print("----------start loading multi_modality -----------")
        self.replace_embedding()

    def replace_embedding(self):
        text_features_list = torch.load(self.args.text_emb_path)
        image_features_list = torch.load(self.args.image_emb_path)
        self.image_mean_embeddings.weight.data[1:-1, :] = image_features_list
        self.image_cov_embeddings.weight.data[1:-1, :] = image_features_list
        self.text_mean_embeddings.weight.data[1:-1, :] = text_features_list
        self.text_cov_embeddings.weight.data[1:-1, :] = text_features_list


    def add_position_mean_embedding(self, sequence, user_interest_embedding):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)

        item_embeddings = self.item_mean_embeddings(sequence)  # (256,100,64)
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))

        fusion_results = self.interaction_expert_layer(item_image_embeddings, item_text_embeddings)
        interaction_emb = self.adaptive_fusion_layer(item_embeddings, fusion_results)
        total_interaction_loss = fusion_results["interaction_losses"]
        
        item_embeddings = item_embeddings + interaction_emb.unsqueeze(1) + item_image_embeddings + item_text_embeddings       



        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb, total_interaction_loss 

    def add_position_cov_embedding(self, sequence, user_interest_embedding):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)

        item_embeddings = self.item_cov_embeddings(sequence)
        item_image_embeddings = self.fc_cov_image(self.image_cov_embeddings(sequence))
        item_text_embeddings = self.fc_cov_text(self.text_cov_embeddings(sequence))

        fusion_results = self.interaction_expert_layer(item_image_embeddings, item_text_embeddings)
        interaction_emb = self.adaptive_fusion_layer(item_embeddings, fusion_results)
        total_interaction_loss = fusion_results["interaction_losses"]
        
        item_embeddings = item_embeddings + interaction_emb.unsqueeze(1) + item_image_embeddings + item_text_embeddings


        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1
        
        return sequence_emb, total_interaction_loss

    def get_user_interest_embedding(self, input_ids, extended_attention_mask):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        preliminary_mean_item_emb = self.item_mean_embeddings(input_ids)
        preliminary_mean_pos_emb = self.position_mean_embeddings(position_ids)
        preliminary_mean_emb = self.LayerNorm(preliminary_mean_item_emb + preliminary_mean_pos_emb)
        preliminary_mean_emb = self.dropout(preliminary_mean_emb)

        preliminary_cov_item_emb = self.item_cov_embeddings(input_ids)
        preliminary_cov_pos_emb = self.position_cov_embeddings(position_ids)
        preliminary_cov_emb = self.LayerNorm(preliminary_cov_item_emb + preliminary_cov_pos_emb)
        preliminary_cov_emb = torch.nn.functional.elu(self.dropout(preliminary_cov_emb)) + 1

        preliminary_encoded_layers = self.interest_encoder(
            preliminary_mean_emb,
            preliminary_cov_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )
        preliminary_output_mean, _, _ = preliminary_encoded_layers[-1]

        user_interest_embedding = preliminary_output_mean[:, -1, :] # Shape: [B, D]
        
        return user_interest_embedding

    def finetune(self, input_ids, user_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        user_interest_embedding = self.get_user_interest_embedding(input_ids, extended_attention_mask)

        mean_sequence_emb, mean_interaction_loss = self.add_position_mean_embedding(input_ids, user_interest_embedding)
        cov_sequence_emb, cov_interaction_loss = self.add_position_cov_embedding(input_ids, user_interest_embedding)
        interaction_loss = (mean_interaction_loss + cov_interaction_loss) / 2.0
        
        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)

        return mean_sequence_output, cov_sequence_output, att_scores, margins, interaction_loss
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




