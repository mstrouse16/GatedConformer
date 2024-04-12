import math
import torch
from torch import nn
import torch.nn.functional as F


class ConvSubSample(nn.Module):
    """
    Convolutional Subsampling

    Subsample the time dimension of the input to reduce dimensionality. This decreases computational expense.

    Apple's paper - "The input first goes through a subsampling convolutional layer
    which reduces its time dimension from 120 to 29"

    Input shape is (batch, features, windows)
    Output shape is (batch, features, windows * reduction factor)

    """

    def __init__(self, subsamplereduction):
        super(ConvSubSample, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, subsamplereduction),
            stride=(1, subsamplereduction),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # set to (batch, 1, features, windows)
        x = self.conv2d(x)  # set to (batch, 1, features, windows/4)
        x = x.squeeze(1)  # set to (batch, features, windows/4)
        return x


class Linear(nn.Module):
    """
    Linear Layer

    Project the feature dimension of the input to increase feature detection.

    Apple's paper - "Its feature dimension also gets projected to the desired hidden size H"

    Input shape is (batch, features, windows * reduction factor)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(self, inputdim, modeldim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(inputdim, modeldim)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  # set to (batch, windows, features) as nn.linear will collapse the left dimensions
        x = self.linear(x)  # Project the feature dimension to a desired size
        return x


class FeedForward(nn.Module):
    """
    Feed Forward Module

    Google's paper - Layernorm, Linear Layer, Swish Activation, Dropout, Linear Layer, Dropout

    Input shape is (batch, windows * reduction factor, model features)
    Hidden shape is (batch, windows * reduction factor, model features * expansion factor)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(self, modeldim, feedforwardexpansion, dropout):
        super(FeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(modeldim)
        self.linear1 = nn.Linear(modeldim, modeldim * feedforwardexpansion)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(modeldim * feedforwardexpansion, modeldim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class Convolution(nn.Module):
    """
    Convolution Module

    Google's paper - Layernorm, Pointwise Conv, Glu Activation, 1D Depthwise Conv, Batchnorm,
                     Swish Activation, Pointwise Conv, Dropout

    Pointwise1 and Pointwise2 applies a sequential kernel which is of size (1, model features)
    Depthwise applies a arbitrary kernel that slides across windows and is seperate for each feature

    Input shape is (batch, windows * reduction factor, model features)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(self, modeldim, pointwiseexpansion, kernel_size, dropout):
        super(Convolution, self).__init__()
        self.layer_norm = nn.LayerNorm(modeldim)
        self.pointwise1 = nn.Conv1d(
            in_channels=modeldim,
            out_channels=modeldim * pointwiseexpansion,
            kernel_size=1,
        )
        self.depthwise = nn.Conv1d(
            in_channels=modeldim,
            out_channels=modeldim,
            kernel_size=kernel_size,
            padding="same",
            groups=modeldim,
        )
        self.batch_norm = nn.BatchNorm1d(modeldim)
        self.pointwise2 = nn.Conv1d(
            in_channels=modeldim, out_channels=modeldim, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(
            0, 2, 1
        )  # set to (batch, features, windows) for sequential calculation
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)  # effectively reduces dim=1 (features) by a factor of 2
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoder(nn.Module):
    """
    Positional Encoder

    Code derived from https://github.com/jreremy/conformer/blob/master/model.py

    Positional encoding based on "Attention Is All You Need"

    Input is len of encodings to get
    Ouput is len modeldim vectors which the positional encodings for each window

    """

    def __init__(self, modeldim, max_length=10000):
        super(PositionalEncoder, self).__init__()
        self.modeldim = modeldim
        encodings = torch.zeros(max_length, modeldim)
        pos = torch.arange(0, max_length, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, modeldim, 2.0) / modeldim))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer("encodings", encodings)

    def forward(self, length):
        return self.encodings[:length, :]


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Self-Attention Module

    Google's paper - Layernorm, Multi-Head Attention with Relative Positional Embedding, Dropout

    Code derived from https://github.com/jreremy/conformer/blob/master/model.py

    Overall method based on "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Input shape is (batch, windows * reduction factor, model features)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(self, modeldim, attentionheads, dropout):
        super(MultiHeadedSelfAttention, self).__init__()
        assert modeldim % attentionheads == 0
        self.modeldim = modeldim
        self.attentionheads = attentionheads
        self.headdim = modeldim // attentionheads

        self.linear_q = nn.Linear(modeldim, modeldim)
        self.linear_k = nn.Linear(modeldim, modeldim)
        self.linear_v = nn.Linear(modeldim, modeldim)
        self.linear_pos = nn.Linear(modeldim, modeldim, bias=False)
        self.linear_final = nn.Linear(modeldim, modeldim)

        self.u = nn.Parameter(torch.Tensor(self.attentionheads, self.headdim))
        self.v = nn.Parameter(torch.Tensor(self.attentionheads, self.headdim))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.layer_norm = nn.LayerNorm(modeldim)
        self.positionalencoder = PositionalEncoder(modeldim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, windows, _ = x.size()
        x = self.layer_norm(x)
        positional_embeddings = self.positionalencoder(windows)
        positional_embeddings = positional_embeddings.repeat(batch_size, 1, 1)

        query = self.linear_q(x).view(
            batch_size, windows, self.attentionheads, self.headdim
        )
        key = (
            self.linear_k(x)
            .view(batch_size, windows, self.attentionheads, self.headdim)
            .permute(0, 2, 3, 1)
        )
        value = (
            self.linear_v(x)
            .view(batch_size, windows, self.attentionheads, self.headdim)
            .permute(0, 2, 3, 1)
        )
        positional_embeddings = (
            self.linear_pos(positional_embeddings)
            .view(batch_size, windows, self.attentionheads, self.headdim)
            .permute(0, 2, 3, 1)
        )

        AC = torch.matmul((query + self.u).transpose(1, 2), key)
        BD = torch.matmul((query + self.v).transpose(1, 2), positional_embeddings)
        BD = self.relative_shift(BD)
        attention = (AC + BD) / math.sqrt(self.modeldim)

        attention = F.softmax(attention, -1)

        x = torch.matmul(attention, value.transpose(2, 3)).transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, self.modeldim)

        x = self.linear_final(x)
        x = self.dropout(x)
        return x

    def relative_shift(self, emb):
        batch_size, attentionheads, length1, length2 = emb.size()
        zeros = emb.new_zeros(batch_size, attentionheads, length1, 1)
        padded = torch.cat([zeros, emb], dim=-1)
        padded = padded.view(batch_size, attentionheads, length2 + 1, length1)
        shifted = padded[:, :, 1:].view_as(emb)
        return shifted


class Gate(nn.Module):
    """
    Binary Gate using the Gumbel-Softmax trick

    Apple's paper - "In our experiments, we use a linear layer to implement each gate."

    Input shape is (batch, windows * reduction factor, model features)
    Output shape is

    """

    def __init__(self, module, modeldim, temperature, gatethreshold, weight=1):
        super(Gate, self).__init__()
        self.module = module
        self.temperature = temperature
        self.gatethreshold = gatethreshold
        self.weight = weight
        self.linear = nn.Linear(modeldim, 2)  # 2 logits for execute and skip

    def forward(self, x):
        x_condensed = x.mean(dim=1)  # Squeeze the time dimension
        logits = self.linear(x_condensed)

        # Calculate probability of execute and skip
        if self.training:
            eps = 1e-20
            U = torch.rand(logits.size(), device=logits.device)
            gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
            logits_and_noise = logits + gumbel_noise
            final_probabilities = F.softmax(logits_and_noise / self.temperature, dim=1)
        else:
            logit_probabilities = F.softmax(logits, dim=1)
            final_probabilities = logit_probabilities

        # Execute or skip
        if self.training:
            output = x + self.weight * self.module(x) * final_probabilities[
                :, 0
            ].unsqueeze(-1).unsqueeze(-1)
        else:  # Batch size will always be 1 at inference as Apple had done
            execute = final_probabilities[:, 0] > self.gatethreshold
            if execute:
                output = x + self.weight * self.module(x)
            else:
                output = x
        return output, final_probabilities[:, 0]


class Block(nn.Module):
    """
    Conformer Block

    Apple's paper - "It then passes through a series of conformer blocks, where each block represents a sequence of
    (i) feedforward, (ii) mutli-headed self attention, (iii) convolution and (iv) feedforward modules"

    Input shape is (batch, windows * reduction factor, model features)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(
        self,
        modeldim,
        feedforwardexpansion,
        feedforwardweight,
        pointwiseexpansion,
        kernel_size,
        attentionheads,
        dropout,
        temperature,
        gatethreshold,
        gates_on,
    ):
        super(Block, self).__init__()

        self.gates_on = gates_on
        self.ffweight = feedforwardweight
        self.ff1 = FeedForward(modeldim, feedforwardexpansion, dropout)
        self.attn = MultiHeadedSelfAttention(modeldim, attentionheads, dropout)
        self.conv = Convolution(modeldim, pointwiseexpansion, kernel_size, dropout)
        self.ff2 = FeedForward(modeldim, feedforwardexpansion, dropout)
        self.layer_norm = nn.LayerNorm(modeldim)

        self.gatedff1 = Gate(
            self.ff1, modeldim, temperature, gatethreshold, feedforwardweight
        )
        self.gatedattn = Gate(self.attn, modeldim, temperature, gatethreshold)
        self.gatedconv = Gate(self.conv, modeldim, temperature, gatethreshold)
        self.gatedff2 = Gate(
            self.ff2, modeldim, temperature, gatethreshold, feedforwardweight
        )

    def forward(self, x):
        gate_values = torch.tensor([]).to(x.device)
        if self.gates_on:
            # Residual connection and weight brought inside the gate
            x, gates = self.gatedff1(x)
            gate_values = torch.cat((gate_values, gates))
            x, gates = self.gatedattn(x)
            gate_values = torch.cat((gate_values, gates))
            x, gates = self.gatedconv(x)
            gate_values = torch.cat((gate_values, gates))
            x, gates = self.gatedff2(x)
            gate_values = torch.cat((gate_values, gates))
        else:
            x = x + (self.ffweight * self.ff1(x))
            x = x + self.attn(x)
            x = x + self.conv(x)
            x = x + (self.ffweight * self.ff2(x))
        x = self.layer_norm(x)

        return x, gate_values


class Conformer(nn.Module):
    """
    Conformer Encoder

    Parameters:
        inputdim  - input dimension of features
        modeldim  - model dimension of features
        feedforwardexpansion - hidden dimension of features in feed forward
        feedforwardweight - weight for feed forward
        pointwiseexpansion - hidden channels for features
        kernel_size - kernel size for 1D depthwise convolution
        blocks - number of conformer blocks to run
        dropout - dropout probability across the model
        temperature - temperature for gumbel softmax
        gatethreshold - threshold for gating
        gates_on - whether gates are turned on or off

    Input shape is (batch, input features, windows)
    Output shape is (batch, windows * reduction factor, model features)

    """

    def __init__(
        self,
        inputdim,
        modeldim,
        subsamplereduction,
        feedforwardexpansion,
        feedforwardweight,
        pointwiseexpansion,
        kernel_size,
        attentionheads,
        blocks,
        dropout,
        temperature,
        gatethreshold,
        gates_on,
    ):
        super(Conformer, self).__init__()
        self.subsample = ConvSubSample(subsamplereduction)
        self.linear = Linear(inputdim, modeldim)
        self.dropout = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    modeldim,
                    feedforwardexpansion,
                    feedforwardweight,
                    pointwiseexpansion,
                    kernel_size,
                    attentionheads,
                    dropout,
                    temperature,
                    gatethreshold,
                    gates_on,
                )
                for i in range(blocks)
            ]
        )

        self.gate_values = None

    def reset_gate_values(self):
        self.gate_values = None

    def enable_gates(self):
        for block in self.blocks:
            block.gates_on = True

    def forward(self, x):
        # Preprocess
        x = self.subsample(x)
        x = self.linear(x)
        x = self.dropout(x)

        if self.gate_values is None:
            self.gate_values = torch.tensor([]).to(x.device)

        # Conformer Blocks
        for block in self.blocks:
            x, gates = block(x)
            self.gate_values = torch.cat((self.gate_values, gates))
        return x


# Mean across time then fc
class EmotionClassifier(nn.Module):
    def __init__(self, conformer, num_labels, modeldim):
        super(EmotionClassifier, self).__init__()
        self.conformer = conformer
        self.linear = nn.Linear(modeldim, num_labels)

    def forward(self, x):
        x = self.conformer(x)
        x = x.mean(dim=1)
        logits = self.linear(x)
        return logits


# Max across time then fc
# class EmotionClassifier(nn.Module):
#     def __init__(self, conformer, num_labels, modeldim):
#         super(EmotionClassifier, self).__init__()
#         self.conformer = conformer
#         self.linear = nn.Linear(modeldim, num_labels)

#     def forward(self, x):
#         x = self.conformer(x)
#         x, _ = x.max(dim=1)  # Global max pooling
#         logits = self.linear(x)
#         return logits


# Max across intervals, mean max then fc
# class EmotionClassifier(nn.Module):
#     def __init__(self, conformer, num_labels, modeldim):
#         super(EmotionClassifier, self).__init__()
#         self.conformer = conformer
#         self.linear = nn.Linear(modeldim, num_labels)
#         self.pool_interval = 10  # Define the interval size

#     def interval_max_pool(self, x, interval):
#         pooled_intervals = []
#         for i in range(0, x.size(1), interval):
#             pooled_interval, _ = x[:, i : i + interval, :].max(dim=1)
#             pooled_intervals.append(pooled_interval)
#         x_pooled = torch.stack(pooled_intervals, dim=1).mean(dim=1)
#         return x_pooled

#     def forward(self, x):
#         x = self.conformer(x)
#         x = self.interval_max_pool(x, self.pool_interval)
#         logits = self.linear(x)
#         return logits


# Statistical pooling with relu
# class EmotionClassifier(nn.Module):
#     def __init__(self, conformer, num_labels, modeldim):
#         super(EmotionClassifier, self).__init__()
#         self.conformer = conformer
#         self.fc1 = nn.Linear(modeldim * 2, modeldim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(modeldim, num_labels)

#     def forward(self, x):
#         x = self.conformer(x)
#         mean = x.mean(dim=1)
#         std = x.std(dim=1)
#         # Concatenate mean and standard deviation
#         concat = torch.cat((mean, std), dim=1)
#         x = self.fc1(concat)
#         x = self.relu(x)
#         logits = self.fc2(x)
#         return logits


class GateLoss(nn.Module):
    def __init__(self, gateregularizer):
        super(GateLoss, self).__init__()
        self.gateregularizer = gateregularizer
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, gate_values):
        base_loss = self.base_loss(outputs, labels)
        gate_loss = gate_values.mean()
        if torch.isnan(gate_loss):
            return base_loss
        else:
            return base_loss + self.gateregularizer * gate_loss
