import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, DistSAEncoder
from modules import LayerNorm, DistSAEncoder
from PRISM import *


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')

        self.has_logged_module_order = False

        self.fc_mean_image = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_mean_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
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


    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))

        fusion_results = self.interaction_expert_layer(item_image_embeddings, item_text_embeddings)
        interaction_emb = self.adaptive_fusion_layer(item_embeddings, fusion_results)
        total_interaction_loss = fusion_results["interaction_losses"]
        
        item_embeddings = item_embeddings + interaction_emb.unsqueeze(1) + item_image_embeddings + item_text_embeddings          

        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb, total_interaction_loss

    def finetune(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_emb, total_interaction_loss = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output, attention_scores = item_encoded_layers[-1]
        return sequence_output, attention_scores, total_interaction_loss

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()