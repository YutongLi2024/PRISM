import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret


def kl_distance(mean1, cov1, mean2, cov2):
    trace_part = torch.sum(cov1 / cov2, -1)
    mean_cov_part = torch.sum((mean2 - mean1) / cov2 * (mean2 - mean1), -1)
    determinant_part = torch.log(torch.prod(cov2, -1) / torch.prod(cov1, -1))

    return (trace_part + mean_cov_part - mean1.shape[1] + determinant_part) / 2


def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, attention_probs


class DistSelfAttention(nn.Module):
    def __init__(self, args):
        super(DistSelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(args.hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.mean_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.cov_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.distance_metric = args.distance_metric
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.gamma = args.kernel_param

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, attention_mask):
        # print("DistSelfAttention forward0")
        mixed_mean_query_layer = self.mean_query(input_mean_tensor)
        mixed_mean_key_layer = self.mean_key(input_mean_tensor)
        mixed_mean_value_layer = self.mean_value(input_mean_tensor)

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor)) + 1
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor)) + 1
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor)) + 1

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        if self.distance_metric == 'wasserstein':
            attention_scores = -wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)
        else:
            attention_scores = -kl_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        mean_context_layer = torch.matmul(attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(attention_probs ** 2, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states, attention_probs


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistIntermediate(nn.Module):
    def __init__(self, args):
        super(DistIntermediate, self).__init__()
        # print("DistIntermediate")
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = nn.ELU()

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        # print("DistIntermediate_forward")
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output, attention_scores


class DistLayer(nn.Module):
    def __init__(self, args):
        super(DistLayer, self).__init__()
        self.attention = DistSelfAttention(args)
        self.mean_intermediate = DistIntermediate(args)
        self.cov_intermediate = DistIntermediate(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask):
        mean_attention_output, cov_attention_output, attention_scores = self.attention(mean_hidden_states, cov_hidden_states, attention_mask)
        mean_intermediate_output = self.mean_intermediate(mean_attention_output)
        cov_intermediate_output = self.activation_func(self.cov_intermediate(cov_attention_output)) + 1
        return mean_intermediate_output, cov_intermediate_output, attention_scores


class DistSAEncoder(nn.Module):
    def __init__(self, args):
        super(DistSAEncoder, self).__init__()
        layer = DistLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            maen_hidden_states, cov_hidden_states, att_scores = layer_module(mean_hidden_states, cov_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        return all_encoder_layers


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([hidden_states, attention_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states, attention_scores])
        return all_encoder_layers


# InDiRec
class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]
        
        :return: A 3d tensor with shape of (N, T_q, C)
        
        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings