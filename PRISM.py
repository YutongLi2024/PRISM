import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ===================================================================
# Expert 
# ===================================================================

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        x = self.drop(x)
        out = self.fc2(out)
        x = self.drop(x)
        return out
    

class InteractionExpertWrapper(nn.Module):
    def __init__(self, expert_model):
        super(InteractionExpertWrapper, self).__init__()
        self.expert_model = expert_model

    def _process_inputs(self, inputs):
        return torch.cat(inputs, dim=-1)

    def _forward_with_replacement(self, inputs, replace_index=None):
        processed_inputs = list(inputs)
        if replace_index is not None:
            random_vector = torch.randn_like(processed_inputs[replace_index])
            processed_inputs[replace_index] = random_vector
        
        x = self._process_inputs(processed_inputs)
        return self.expert_model(x)    

    def forward(self, inputs):
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_multiple(self, inputs):
        outputs = []
        outputs.append(self.forward(inputs))
        for i in range(len(inputs)):
            outputs.append(self._forward_with_replacement(inputs, replace_index=i))
        return outputs


# ===================================================================
# Interaction Expert Layer
# ===================================================================

class Interaction_Expert_Layer(nn.Module):
    def __init__(self, args, hidden_size, expert_hidden_size, num_experts_per_type=1):
        super().__init__()
        self.num_modalities = 2  # img_feat, txt_feat
        self.num_branches = self.num_modalities * num_experts_per_type + 2
        self.interaction_experts = nn.ModuleList()
        for _ in range(self.num_branches):
            base_expert = Expert(
                input_size=hidden_size * self.num_modalities,
                output_size=hidden_size,
                hidden_size=expert_hidden_size,
            )
            self.interaction_experts.append(InteractionExpertWrapper(deepcopy(base_expert)))
        self.lambda_uni_v = 0.1
        self.lambda_uni_t = 0.1
        self.lambda_syn = 0.1 
        self.lambda_red = 0.1
        
    def uniqueness_loss_single(self, anchor, pos, neg, margin=1.0):
        # Triplet loss for uniqueness
        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        # Pushes anchor away from all single-modality-missing representations
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=-1)
            cosine_sim = torch.einsum('bd,bd->b', anchor_normalized, negative_normalized)
            total_syn_loss += cosine_sim.mean()
        return total_syn_loss / len(negatives)

    def redundancy_loss(self, anchor, positives):
        # Pulls anchor towards all single-modality-missing representations
        total_red_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=-1)
            cosine_sim = torch.einsum('bd,bd->b', anchor_normalized, positive_normalized)
            total_red_loss += (1 - cosine_sim).mean()
        return total_red_loss / len(positives)
    
    def calculate_total_interaction_loss(self, interaction_losses_dict):
                
        total_loss = 0.0
        
        if interaction_losses_dict is None:
            return total_loss

        uni_v_loss = interaction_losses_dict.get("uniqueness_v")
        if uni_v_loss is not None:
            total_loss += self.lambda_uni_v * uni_v_loss

        uni_t_loss = interaction_losses_dict.get("uniqueness_t")
        if uni_t_loss is not None:
            total_loss += self.lambda_uni_t * uni_t_loss
                
        synergy_loss = interaction_losses_dict.get("synergy")
        if synergy_loss is not None:
            total_loss += self.lambda_syn * synergy_loss

        redundancy_loss = interaction_losses_dict.get("redundancy")
        if redundancy_loss is not None:
            total_loss += self.lambda_red * redundancy_loss
            
        return total_loss


    def forward(self, img_feat, txt_feat):
        if img_feat.dim() == 3:
            img_feat_proc = img_feat.mean(dim=1)
            txt_feat_proc = txt_feat.mean(dim=1)
        else:
            img_feat_proc = img_feat
            txt_feat_proc = txt_feat

        inputs = [img_feat_proc, txt_feat_proc]

        expert_outputs = []
        for expert in self.interaction_experts:
            expert_outputs.append(expert.forward_multiple(inputs))

        # uniqueness_losses
        uniqueness_losses = []
        for i in range(self.num_modalities):
            outputs = expert_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]
            positives = outputs[1:i+1] + outputs[i+2:]
            uni_loss_i = 0
            for pos in positives:
                uni_loss_i += self.uniqueness_loss_single(anchor, pos, neg)
            uniqueness_losses.append(uni_loss_i / len(positives))

        # synergy loss
        synergy_outputs = expert_outputs[-2]
        syn_loss = self.synergy_loss(synergy_outputs[0], synergy_outputs[1:])

        # redundancy loss
        redundancy_outputs = expert_outputs[-1]
        red_loss = self.redundancy_loss(redundancy_outputs[0], redundancy_outputs[1:])

        interaction_losses_dict = {
            "uniqueness_v": uniqueness_losses[0],
            "uniqueness_t": uniqueness_losses[1],
            "synergy": syn_loss,
            "redundancy": red_loss,
        }

        expert_embs = {
            "uni_v": expert_outputs[0][0],
            "uni_t": expert_outputs[1][0],
            "syn": expert_outputs[-2][0],
            "rdn": expert_outputs[-1][0],
        }

        total_interaction_loss = self.calculate_total_interaction_loss(interaction_losses_dict)

        return {
            "interaction_losses": total_interaction_loss,
            "expert_embs": expert_embs,
        }



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPReWeighting(nn.Module):
    def __init__(self, num_modalities, num_branches, hidden_dim, hidden_dim_rw=128, num_layers=2, temperature=1.0):
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.GELU(),
            dropout=0.3,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1) # Shape: [B, num_modalities * D]
        
        logits = self.mlp(x) # Shape: [B, num_branches]
        weights = self.temperature_scaled_softmax(logits) # Shape: [B, num_branches]
        return weights


class Adaptive_Fusion_Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.adaptive_fusion_mlp = MLPReWeighting(
            num_modalities = 5,
            num_branches = 4,
            hidden_dim = args.hidden_size,
            hidden_dim_rw = args.hidden_size * 2,
            num_layers = 2,
            temperature = getattr(args, 'temperature', 0.7)
        )

    def forward(self, item_embeddings, fusion_results):

        expert_embs = fusion_results["expert_embs"]
        id_emb = item_embeddings.mean(dim=1)

        gating_inputs = [
            expert_embs["uni_v"],
            expert_embs["uni_t"],
            expert_embs["syn"],
            expert_embs["rdn"],
            id_emb
        ]

        weights = self.adaptive_fusion_mlp(gating_inputs)


        interaction_emb = (
            weights[:,0:1] * expert_embs["uni_v"] +
            weights[:,1:2] * expert_embs["uni_t"] +
            weights[:,2:3] * expert_embs["syn"] +
            weights[:,3:4] * expert_embs["rdn"]
        )

        return interaction_emb
