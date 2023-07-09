"""Modified BERT model with conditional output layer run iteratively. See doc/report.pdf for details."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import os
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling import \
    BertModel, BertConfig, BertSelfAttention, \
    BertIntermediate, BertOutput, BertLayerNorm, BertLayer, BertEncoder
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


logger = logging.getLogger(__name__)


class OutputLayerL3(nn.Module):
    """Output layer with three types of losses: original SQuAD loss, token-level loss, and has-answer loss."""

    def __init__(self, use_tl, use_ha, hidden_size, num_attention_heads=8, aux_loss_weight=0.1, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.aux_loss_weight = aux_loss_weight
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.use_tl = use_tl
        self.use_ha = use_ha

        # QA output layers
        self.qa_start_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.qa_relu = nn.ReLU()
        self.qa_start_dropout = nn.Dropout(self.dropout_prob)
        self.qa_start_outputs = nn.Linear(self.hidden_size, 1)
        self.qa_end_hidden = nn.Linear(self.hidden_size + self.hidden_size + 1, self.hidden_size)
        self.qa_end_dropout = nn.Dropout(self.dropout_prob)
        self.qa_end_outputs = nn.Linear(self.hidden_size, 1)

        # Token label output layers
        if self.use_tl:
            self.tl_outputs_hidden = nn.Linear(self.hidden_size, self.hidden_size)
            self.tl_outputs_relu = nn.ReLU()
            self.tl_dropout = nn.Dropout(self.dropout_prob)
            self.tl_outputs = nn.Linear(self.hidden_size, 1)

        # Has-Answer output layers
        if self.use_ha:
            self.ha_att = SelfAttention(self.hidden_size, self.num_attention_heads, dropout_prob=dropout_prob)
            self.ha_outputs_hidden = nn.Linear(self.hidden_size, self.hidden_size)
            self.ha_outputs_relu = nn.ReLU()
            self.ha_dropout = nn.Dropout(self.dropout_prob)
            self.ha_outputs = nn.Linear(self.hidden_size, 1)

    def forward(self, sequence_output, attention_mask,
                start_positions=None, end_positions=None, token_labels=None, has_answers=None):
        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)

        # QA output
        start_h = self.qa_start_hidden(sequence_output)
        start_h = self.qa_relu(start_h)  # (B, S, H)
        start_h_d = self.qa_start_dropout(start_h)
        start_logits_unsq = self.qa_start_outputs(start_h_d)  # (B, S, 1)
        start_logits = start_logits_unsq.view(batch_size, -1)  # squeeze(-1)  # (B, S)

        end_h = self.qa_end_hidden(torch.cat((sequence_output, start_h, start_logits_unsq), dim=-1))
        end_h = self.qa_relu(end_h)
        end_h_d = self.qa_end_dropout(end_h)
        end_logits = self.qa_end_outputs(end_h_d).view(batch_size, -1)  # squeeze(-1)  # (B, S)

        # Token label output
        if self.use_tl:
            tl_h = self.tl_outputs_hidden(sequence_output)
            tl_h = self.tl_outputs_relu(tl_h)
            tl_h = self.tl_dropout(tl_h)
            tl_logits = self.tl_outputs(tl_h).view(batch_size, -1)  # squeeze(-1)  # (B, S)
        else:
            tl_logits = torch.zeros(batch_size, seq_len).detach().cuda()  # (B, S), dummy

        # Has-Answer output
        if self.use_ha:
            ha_h = self.ha_att(sequence_output, attention_mask)  # (B, H)
            ha_h = self.ha_outputs_hidden(ha_h)  # (B, H)
            ha_h = self.ha_outputs_relu(ha_h)
            ha_h = self.ha_dropout(ha_h)
            ha_logits = self.ha_outputs(ha_h).view(batch_size)  # squeeze(-1)  # (B,)
        else:
            ha_logits = torch.zeros(batch_size).detach().cuda()  # (B,), dummy

        total_loss = None
        if start_positions is not None and end_positions is not None \
                and token_labels is not None and has_answers is not None:
            # QA output
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # # Token label output
            tl_loss = 0
            if self.use_tl:
                tl_losses = []
                batch_size = tl_logits.size(0)
                for i in range(batch_size):
                    mask = attention_mask[i].to(torch.float32)
                    mask[0] = 1.
                    tl_bcel = nn.BCEWithLogitsLoss(weight=mask, reduction='mean')
                    tl_losses.append(tl_bcel(tl_logits[i], token_labels[i].to(torch.float32)))
                tl_loss = torch.stack(tl_losses).mean()

            # Has-Answer output
            ha_loss = 0
            if self.use_ha:
                ha_bcel = nn.BCEWithLogitsLoss(reduction='mean')
                ha_loss = ha_bcel(ha_logits, has_answers.to(torch.float32))

            # Sum of loss: original SQuAD loss and losses from aux tasks
            total_loss = (start_loss + end_loss) * 0.5 + self.aux_loss_weight * (tl_loss + ha_loss)

        return start_logits, end_logits, tl_logits, ha_logits, total_loss


class ModifiedBertSelfAttention(BertSelfAttention):
    """Same as BertSelfAttention but using input_config.hidden_size for query, key, value."""
    def __init__(self, input_config, config):
        super().__init__(config)

        self.query = nn.Linear(input_config.hidden_size, self.all_head_size)
        self.key = nn.Linear(input_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(input_config.hidden_size, self.all_head_size)


class ModifiedBertLayer(nn.Module):
    """Same as BertLayer but using ModifiedBertSelfAttention as the attention module."""

    def __init__(self, input_config, config):
        super().__init__()
        # MODIFIED: To incorporate the augmented input dims.
        self.attention = ModifiedBertSelfAttention(input_config, config)
        # MODIFIED_END
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        # MODIFIED: Extend attention mask to accommodate extra dims for conditional inputs.
        extended_attention_mask = attention_mask.to(torch.float32).unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # MODIFIED_END
        attention_output = self.attention(hidden_states, extended_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ModifiedBertEncoder(nn.Module):
    """BertEncoder with conditional inputs."""

    def __init__(self, input_config: BertConfig, config: BertConfig):
        """Constructor.
        
        Args:
            input_config: config with hidden size augmented due to conditional inputs.
            config: Original configuration for Bert.
        """
        super().__init__()
        layers = [ModifiedBertLayer(input_config, config)]  # First layer: with conditions as input
        # TODO: Currently, num_hidden_layers >= 2 does not work. Need to know why
        for _ in range(config.num_hidden_layers - 1):
            layers.append(
                ModifiedBertLayer(config, config)
            )
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ConditionalLayer(nn.Module):
    """Conditional layer: Bert encoder with augmented inputs and outputs with aux losses."""

    def __init__(self,
                 num_losses: int = 4,
                 hidden_size: int = 320,
                 num_hidden_layers: int = 1,
                 num_attention_heads: int =4 ,
                 intermediate_size: int = 1500,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 out_aux_loss_weight: float = 0.1,
                 out_dropout_prob: float = 0.2,
                 out_num_attention_heads: int = 6,
                 use_tl: bool = True,
                 use_ha: bool = True):
        """Constructor.
        
        Args:
            num_losses: number of loss functions. Note that start_idx and end_idx predictions count as 2.
            hidden_size: hidden size for attention.
            num_hidden_layers: number of encoder layer.
            num_attention_heads: number of attention heads.
            intermediate_size: feed-forward size.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
            out_aux_loss_weight: Relative loss weight for aux tasks with respect to the standard
                start_idx and end_idx prediction losses.
            out_dropout_prob: Dropout rate for output layer.
            out_num_attention_heads: Number attention heads for output layer.
            use_tl: Whether to use token-level loss or not.
            use_ha: Whether to use has-answer loss or not.
        """
        super().__init__()
        self.config = BertConfig(
            0,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.input_config = copy.deepcopy(self.config)
        self.input_config.hidden_size += num_losses
        self.encoder = ModifiedBertEncoder(self.input_config, self.config)
        self.output = OutputLayerL3(
            use_tl,
            use_ha,
            hidden_size,
            num_attention_heads=out_num_attention_heads,
            aux_loss_weight=out_aux_loss_weight,
            dropout_prob=out_dropout_prob
        )

    def forward(self, hidden_states, conditions, attention_mask,
                start_positions=None, end_positions=None, token_labels=None, has_answers=None):
        hidden_states = torch.cat((hidden_states, conditions), dim=-1)
        hidden_states = self.encoder(hidden_states, attention_mask, output_all_encoded_layers=False)
        outputs = self.output(hidden_states[-1], attention_mask,
                              start_positions=start_positions, end_positions=end_positions,
                              token_labels=token_labels, has_answers=has_answers)

        start_logits, end_logits, tl_logits, ha_logits, loss = outputs
        start_logits = start_logits.unsqueeze(-1)
        end_logits = end_logits.unsqueeze(-1)
        tl_logits = tl_logits.unsqueeze(-1)
        ha_logits = ha_logits.unsqueeze(-1).repeat((1, start_logits.size(1))).unsqueeze(-1)

        conditions = torch.cat((start_logits, end_logits, tl_logits, ha_logits), dim=-1)

        return hidden_states[-1], conditions, loss


class ConditionalRefinement(nn.Module):
    """The overall Conditional Refinement Network as described in doc/report.pdf."""

    def __init__(
        self,
        seq_len: int = 448,
        num_losses: int = 4,
        hidden_size: int = 320,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 4,
        intermediate_size: int = 1500,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        out_aux_loss_weight: float = 0.1,
        out_dropout_prob: float = 0.2,
        out_num_attention_heads: int = 6,
        use_tl: bool = True,
        use_ha: bool = True,
    ):
        """Constructor.
        
        Args:
            seq_len: Max sequence length for the encoder. Also used for initializing the conditional input.
            num_losses: number of loss functions. Note that start_idx and end_idx predictions count as 2.
            hidden_size: hidden size for attention.
            num_hidden_layers: number of encoder layer.
            num_attention_heads: number of attention heads.
            intermediate_size: feed-forward size.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
            out_aux_loss_weight: Relative loss weight for aux tasks with respect to the standard
                start_idx and end_idx prediction losses.
            out_dropout_prob: Dropout rate for output layer.
            out_num_attention_heads: Number attention heads for output layer.
            use_tl: Whether to use token-level loss or not.
            use_ha: Whether to use has-answer loss or not.
        """
        super().__init__()

        # TODO: independent conditional layers. Need to investigate whether it is better than iterative version
        # self.num_iters = num_iters
        # self.loss_ratio = loss_ratio
        # clayers = []
        # for _ in range(num_iters):
        #     clayers.append(
        #         ConditionalLayer(
        #             num_losses=num_losses,
        #             hidden_size=hidden_size,
        #             num_hidden_layers=num_hidden_layers,
        #             num_attention_heads=num_attention_heads,
        #             intermediate_size=intermediate_size,
        #             hidden_dropout_prob=hidden_dropout_prob,
        #             attention_probs_dropout_prob=attention_probs_dropout_prob,
        #             out_aux_loss_weight=out_aux_loss_weight,
        #             out_dropout_prob=out_dropout_prob,
        #             out_num_attention_heads=out_num_attention_heads
        #         )
        #     )
        # self.clayer = nn.ModuleList(clayers)
        self.clayer = ConditionalLayer(
            num_losses=num_losses,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            out_aux_loss_weight=out_aux_loss_weight,
            out_dropout_prob=out_dropout_prob,
            out_num_attention_heads=out_num_attention_heads,
            use_tl=use_tl,
            use_ha=use_ha
        )
        self.ic = torch.nn.Parameter(torch.zeros(1, seq_len, num_losses))  # four losses

        # TODO: Alternative conditions. Not 100% working. Need to understand why
        # self.ic_qa = torch.nn.Parameter(torch.zeros(1, seq_len, 2).cuda())
        # if use_tl:
        #     self.ic_tl = torch.nn.Parameter(torch.zeros(1, seq_len, 1).cuda())
        # else:
        #     self.ic_tl = torch.zeros(1, seq_len, 1).cuda()
        #
        # if use_ha:
        #     self.ic_ha = torch.nn.Parameter(torch.zeros(1, seq_len, 1).cuda())
        # else:
        #     self.ic_ha = torch.zeros(1, seq_len, 1).cuda()

    @classmethod
    def conditions2logits(cls, conditions, batch_size):
        start_logits = conditions[:, :, 0].view(batch_size, -1)  # .squeeze(-1)
        end_logits = conditions[:, :, 1].view(batch_size, -1)  # .squeeze(-1)
        tl_logits = conditions[:, :, 2].view(batch_size, -1)  # .squeeze(-1)
        ha_logits = conditions[:, 0, 3].view(batch_size)  # .squeeze(-1).squeeze(-1)
        return start_logits, end_logits, tl_logits, ha_logits

    @classmethod
    def conditions2logits2(cls, conditions, batch_size):
        start_logits = conditions[:, :, 0].view(batch_size, -1)  # .squeeze(-1)
        end_logits = conditions[:, :, 1].view(batch_size, -1)  # .squeeze(-1)
        return start_logits, end_logits

    @classmethod
    def inputs2conditions2(cls, start_positions, end_positions, batch_size, seq_len):
        start_logits = torch.zeros(batch_size, seq_len)
        end_logits = torch.zeros(batch_size, seq_len)

        def set_positions(logits, positions):
            for i in range(batch_size):
                logits[i, positions[i]] = 1

        set_positions(start_logits, start_positions)
        set_positions(end_logits, end_positions)
        return torch.stack((start_logits, end_logits), dim=-1).cuda().detach()

    def forward(self, hidden_states, attention_mask,
                start_positions=None, end_positions=None, token_labels=None, has_answers=None,
                num_iters=3):
        outputs = []
        batch_size = hidden_states.size(0)
        # TODO: alternative ic as concat of qa, tl, and ha. not 100% working. Not sure why
        # ic = torch.cat((self.ic_qa, self.ic_tl, self.ic_ha), dim=-1)
        conditions = self.ic.expand((batch_size, -1, -1))
        for i in range(num_iters):  # num of refinements
            hidden_states, conditions, loss = self.clayer(
                hidden_states, conditions, attention_mask,
                start_positions=start_positions, end_positions=end_positions,
                token_labels=token_labels, has_answers=has_answers
            )
            outputs.append(self.conditions2logits(conditions, batch_size))
        # TODO: independent conditional layers. Need to investigate whether it is better than iterative version
        # for i in range(self.num_iters):
        #     if i == 0:
        #         hidden_states, conditions, loss = self.clayer[i](
        #             hidden_states, inputs, attention_mask,
        #             start_positions=start_positions, end_positions=end_positions,
        #             token_labels=token_labels, has_answers=has_answers
        #         )
        #         inputs = self.inputs2conditions2(start_positions, end_positions, batch_size, seq_len)
        #     else:
        #         hidden_states, conditions, loss = self.clayer[i](
        #             hidden_states, inputs, attention_mask,
        #             start_positions=start_positions, end_positions=end_positions,
        #             token_labels=token_labels, has_answers=has_answers
        #         )
        #     # outputs.append(self.conditions2logits(conditions, batch_size))
        #     outputs.append(self.conditions2logits2(conditions, batch_size))
        #     losses.append(loss * self.loss_ratio**(self.num_iters-i-1))
        #     # losses.append(loss)

        if start_positions is not None and end_positions is not None \
                and token_labels is not None and has_answers is not None:
            total_loss = loss
        else:
            total_loss = None

        # TODO: multiple losses, each for each refinement layer. Does not work at this moment
        # if start_positions is not None and end_positions is not None \
        #         and token_labels is not None and has_answers is not None:
            # total_loss = torch.tensor(losses, requires_grad=True).mean()
            # total_loss.append(losses)
        return outputs, total_loss


class Adaptor(nn.Module):
    """Simple Dense-Dropout-LayerNorm layer to adapt from `in_size` to `out_size`."""

    def __init__(self, in_size: int, out_size: int, dropout_prob: float = 0.2):
        """Constructor.
        
        Args:
            in_size: input size for adapting from.
            out_size: output size for adapting to.
            dropout_prob: drop out rate.
        """
        super().__init__()
        self.dense = nn.Linear(in_size, out_size)
        self.LayerNorm = BertLayerNorm(out_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertQAL3(nn.Module):
    """End-to-end BertQA model with conditional iterative refinement layers."""
    def __init__(self,
                 enc_bert_model: str,
                 enc_selected_layers: List[int] = None,
                 cir_seq_len: int = 448,
                 cir_num_losses: int = 4,
                 cir_hidden_size: int = 320,
                 cir_num_hidden_layers: int = 1,
                 cir_num_attention_heads: int = 4,
                 cir_intermediate_size: int = 1280,
                 cir_hidden_dropout_prob: float = 0.1,
                 cir_attention_probs_dropout_prob: float = 0.1,
                 out_num_attention_heads: int = 8,
                 out_aux_loss_weight: float = 0.1,
                 out_dropout_prob: float = 0.2,
                 use_tl = True,
                 use_ha = True
                 ):
        """Constructor.
        
        Args:
            enc_bert_model: model name to load from pretrained models.
            enc_selected_layers: selected hidden layers will be adapted and fed to the Conditional
                Refinement network.
            cir_seq_len: Sequence length for conditional iterative refinement.
            cir_num_losses: Number of losses. Note that start_idx and end_idx count as two.
            cir_hidden_size: Hidden size dims.
            cir_num_hidden_layers: Number of encoder layers.
            cir_num_attention_heads: Number of attention heads.
            cir_intermediate_size: Feed-forward layer size.
            cir_hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            cir_attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
            out_num_attention_heads: Number attention heads for output layer.
            out_aux_loss_weight: Relative loss weight for aux tasks with respect to the standard
                start_idx and end_idx prediction losses.
            out_dropout_prob: Dropout rate for output layer.
            use_tl: Whether to use token-level loss or not.
            use_ha: Whether to use has-answer loss or not.
        """
        super().__init__()
        # Encoder
        self.enc_bert = BertModel.from_pretrained(
                enc_bert_model,
                cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        )
        self.enc_selected_layers = enc_selected_layers
        if not self.enc_selected_layers:
            self.enc_output_size = self.enc_bert.config.hidden_size
        else:
            self.enc_output_size = self.enc_bert.config.hidden_size * len(self.enc_selected_layers)
        self.enc_hidden_size = self.enc_bert.config.hidden_size

        # CIR
        self.cir_seq_len = cir_seq_len
        self.cir_num_losses = cir_num_losses
        self.cir_hidden_size = cir_hidden_size
        self.cir_num_hidden_layers = cir_num_hidden_layers
        self.cir_num_attention_heads = cir_num_attention_heads
        self.cir_intermediate_size = cir_intermediate_size
        self.cir_hidden_dropout_prob = cir_hidden_dropout_prob
        self.cir_attention_probs_dropout_prob = cir_attention_probs_dropout_prob

        # Output
        self.out_num_attention_heads = out_num_attention_heads
        self.out_aux_loss_weight = out_aux_loss_weight
        self.out_dropout_prob = out_dropout_prob

        # INIT
        self.adaptor = Adaptor(self.enc_hidden_size * len(self.enc_selected_layers), self.cir_hidden_size)
        self.cir = ConditionalRefinement(
            seq_len=self.cir_seq_len,
            num_losses=self.cir_num_losses,
            hidden_size=self.cir_hidden_size,
            num_hidden_layers=self.cir_num_hidden_layers,
            num_attention_heads=self.cir_num_attention_heads,
            intermediate_size=self.cir_intermediate_size,
            hidden_dropout_prob=self.cir_hidden_dropout_prob,
            attention_probs_dropout_prob=self.cir_attention_probs_dropout_prob,
            out_aux_loss_weight=self.out_aux_loss_weight,
            out_dropout_prob=self.out_dropout_prob,
            out_num_attention_heads=self.out_num_attention_heads,
            use_tl=use_tl,
            use_ha=use_ha
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                token_labels=None, has_answers=None, num_iters=1):

        seq_out, _ = self.enc_bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)  # (B, S, H*L)
        x = torch.cat([seq_out[x] for x in self.enc_selected_layers], dim=-1)  # (B, S, H*ML)

        x = self.adaptor(x)  # (B, S, CH), CH: cir_hidden_size

        outputs, total_loss = self.cir(
            x, attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            token_labels=token_labels,
            has_answers=has_answers,
            num_iters=num_iters)

        return outputs, total_loss


class SelfAttention(nn.Module):
    """Custom self attention layer: use a input-independent trainable vector `u` to attend."""

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        """Constructor.
        
        Args:
            hidden_size: Hidden size dim.
            num_attention_heads: Number of attention heads.
            dropout_prob: Drop-out rate.
        """
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

        self.u = torch.nn.Parameter(torch.randn(self.attention_head_size, requires_grad=True))

    def transpose_for_scores(self, x):
        # x shape: (B, S, H)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (B, S, A, HS)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (B, A, S, HS)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  # (B, S, H)
        mixed_value_layer = self.value(hidden_states)  # (B, S, H)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (B, A, S, HS)
        key_layer = self.u                                          # (HS,)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (B, A, S, HS)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)  # (B, A, S, HS) x (HS,) -> (B, A, S)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (B, A, S)
        extended_attention_mask = attention_mask.to(torch.float32).unsqueeze(1)  # (B, 1, S)
        attention_scores = attention_scores + (1.0 - extended_attention_mask) * -10000.0  # (B, A, S)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (B, A, S), att on last dim

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs.unsqueeze(2), value_layer).squeeze(2)
        # (B, A, 1, S) x (B, A, S, HS) -> (B, A, 1, HS) -> (B, A, HS)
        new_context_layer_shape = (context_layer.size(0), self.all_head_size)  # (B, H)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
