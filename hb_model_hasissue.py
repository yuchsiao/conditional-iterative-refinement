from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling import \
    BertPreTrainedModel, BertModel, BertEncoder, BertConfig, \
    BertAttention, BertIntermediate, BertOutput, BertLayerNorm, \
    ACT2FN, BertLayer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


logger = logging.getLogger(__name__)


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
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
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertQAFixed(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)
        # Fix BERT model
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                prerun=False):
        # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        if prerun:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            return sequence_output

        logits = self.qa_outputs(input_ids)
        # logger.info("logits size: {}".format(logits.size()))
        start_logits, end_logits = logits.split(1, dim=-1)
        # logger.info("start logits size: {}".format(start_logits.size()))
        # logger.info("end logits size: {}".format(end_logits.size()))
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # logger.info("start logits size: {}".format(start_logits.size()))
        # logger.info("end logits size: {}".format(end_logits.size()))

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            # logger.info("ignored_index: {}".format(ignored_index))
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            # logger.info("start loss: {}".format(start_loss))
            end_loss = loss_fct(end_logits, end_positions)
            # logger.info("end loss: {}".format(end_loss))
            total_loss = (start_loss + end_loss) / 2
            return start_logits, end_logits, total_loss
        else:
            return start_logits, end_logits


# class BertQA(BertPreTrainedModel):
#     """BERT model for Question Answering (span extraction).
#     This module is composed of the BERT model with a linear layer on top of
#     the sequence output that computes start_logits and end_logits
#
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#         `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#
#     Outputs:
#         if `start_positions` and `end_positions` are not `None`:
#             Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
#         if `start_positions` or `end_positions` is `None`:
#             Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
#             position tokens of shape [batch_size, sequence_length].
#
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#
#     model = BertForQuestionAnswering(config)
#     start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.apply(self.init_bert_weights)
#         # Fix BERT model
#         # for param in self.bert.parameters():
#         #     param.requires_grad = False
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
#                 prerun=False):
#         # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         logits = self.qa_outputs(sequence_output)
#         # logger.info("logits size: {}".format(logits.size()))
#         start_logits, end_logits = logits.split(1, dim=-1)
#         # logger.info("start logits size: {}".format(start_logits.size()))
#         # logger.info("end logits size: {}".format(end_logits.size()))
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#         # logger.info("start logits size: {}".format(start_logits.size()))
#         # logger.info("end logits size: {}".format(end_logits.size()))
#
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#             # logger.info("ignored_index: {}".format(ignored_index))
#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             # logger.info("start loss: {}".format(start_loss))
#             end_loss = loss_fct(end_logits, end_positions)
#             # logger.info("end loss: {}".format(end_loss))
#             total_loss = (start_loss + end_loss) / 2
#             return start_logits, end_logits, total_loss
#         else:
#             return start_logits, end_logits


# class BertQA(nn.Module):
#     """BERT model for Question Answering (span extraction).
#     This module is composed of the BERT model with a linear layer on top of
#     the sequence output that computes start_logits and end_logits
#
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#         `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
#             Positions are clamped to the length of the sequence and position outside of the sequence are not taken
#             into account for computing the loss.
#
#     Outputs:
#         if `start_positions` and `end_positions` are not `None`:
#             Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
#         if `start_positions` or `end_positions` is `None`:
#             Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
#             position tokens of shape [batch_size, sequence_length].
#
#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#
#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#
#     model = BertForQuestionAnswering(config)
#     start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, bert_model, tune_k=0):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(
#                 bert_model,
#                 cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
#         )
#         self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
#         # self.apply(self.init_bert_weights)
#
#         # Decompose last k layers
#         if tune_k > 0:
#             layers = []
#             for i in range(-tune_k, 0):
#                 layers.append(self.bert._modules["encoder"]._modules["layer"][i])
#             assert len(layers) == tune_k
#             original_k = len(self.bert._modules['encoder']._modules['layer'])
#             self.tf_layers = nn.ModuleList(layers)
#             self.bert._modules["encoder"]._modules["layer"] = self.bert._modules["encoder"]._modules["layer"][:-tune_k]
#             assert len(self.bert._modules['encoder']._modules['layer']) + tune_k == original_k, \
#                     "{} + {} != {}".format(len(self.bert._modules['encoder']._modules['layer']), tune_k, original_k)
#
#         # Fix BERT model
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
#                 prerun=False):
#         # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         if prerun:
#             sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#             return sequence_output
#
#         hidden_states = input_ids
#         attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.float32)
#         for layer in self.tf_layers:
#             hidden_states = layer(hidden_states, attention_mask)
#
#         logits = self.qa_outputs(hidden_states)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             return start_logits, end_logits, total_loss
#         else:
#             return start_logits, end_logits

class BertQA(nn.Module):
    def __init__(self, bert_model: str, tune_k: int=0, selected_layers: List[int]=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(
                bert_model,
                cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        )
        self.selected_layers = selected_layers
        if not self.selected_layers:
            self.qa_outputs_hidden = nn.Linear(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size)
            self.qa_outputs_relu = nn.ReLU()
            self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        else:
            self.qa_outputs_hidden = nn.Linear(
                self.bert.config.hidden_size * len(self.selected_layers),
                self.bert.config.hidden_size)
            self.qa_outputs_relu = nn.ReLU()
            self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)

        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                token_labels=None, has_answers=None,
                prerun=False):

        if not self.selected_layers:
            sequence_output, _ = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)  # (B, T, H)
        else:
            seq_out, _ = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)  # (B, T, H*L)
            sequence_output = torch.cat([seq_out[x] for x in self.selected_layers], dim=-1)  # (B, T, H*ML)

        hiddens = self.qa_outputs_hidden(sequence_output)
        hiddens = self.qa_outputs_relu(hiddens)
        logits = self.qa_outputs(hiddens)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
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
            total_loss = (start_loss + end_loss) / 2
            return start_logits, end_logits, total_loss
        else:
            return start_logits, end_logits


class OutputLayerL3(nn.Module):
    def __init__(self, use_tl, use_ha, hidden_size, num_attention_heads=8, aux_loss_weight=0.1, dropout_prob=0.1):
        super().__init__()
        # self.seq_output_size = seq_output_size
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

            # Sum of loss
            # total_loss = [start_loss, end_loss, tl_losses, ha_loss]
            total_loss = (start_loss + end_loss) * 0.5 + self.aux_loss_weight * (tl_loss + ha_loss)
            # total_loss = (start_loss + end_loss) * 0.5 + self.aux_loss_weight * ha_loss
            # total_loss = (start_loss + end_loss) * 0.5  # + self.aux_loss_weight * ha_loss

        return start_logits, end_logits, tl_logits, ha_logits, total_loss
        # return start_logits, end_logits, total_loss
        # return (start_logits, end_logits, tl_logits, ha_logits), total_loss


class ModifiedBertSelfAttention(nn.Module):
    def __init__(self, input_config, config):
        super(ModifiedBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_config.hidden_size, self.all_head_size)
        self.key = nn.Linear(input_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(input_config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class ModifiedBertIntermediate(nn.Module):
    def __init__(self, input_config, config):
        super(ModifiedBertIntermediate, self).__init__()
        self.dense = nn.Linear(input_config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModifiedBertLayer(nn.Module):
    def __init__(self, input_config, config):
        super(ModifiedBertLayer, self).__init__()
        self.attention = ModifiedBertSelfAttention(input_config, config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = attention_mask.to(torch.float32).unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_output = self.attention(hidden_states, extended_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ModifiedBertEncoder(nn.Module):
    def __init__(self, input_config, config):
        super(ModifiedBertEncoder, self).__init__()
        layers = [ModifiedBertLayer(input_config, config)]
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
    def __init__(self,
                 num_losses=4,
                 hidden_size=320,
                 num_hidden_layers=1,
                 num_attention_heads=4,
                 intermediate_size=1500,
                 # hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 # max_position_embeddings=512,
                 # type_vocab_size=2,
                 # initializer_range=0.02,
                 out_aux_loss_weight=0.1,
                 out_dropout_prob=0.2,
                 out_num_attention_heads=6,
                 use_tl=True,
                 use_ha=True):
        super().__init__()
        self.config = BertConfig(
            0,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            # hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            # max_position_embeddings=max_position_embeddings,
            # type_vocab_size=type_vocab_size,
            # initializer_range=initializer_range
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

        # start_logits, end_logits, loss = outputs
        # start_logits = start_logits.unsqueeze(-1)
        # end_logits = end_logits.unsqueeze(-1)
        # conditions = torch.cat((start_logits, end_logits), dim=-1)

        # logger.info("loss: {}".format(loss))
        return hidden_states[-1], conditions, loss


class ConditionalRefinement(nn.Module):
    def __init__(self,
                 # num_iters=3,
                 # loss_ratio=0.5,
                 seq_len=448,
                 num_losses=4,
                 hidden_size=320,
                 num_hidden_layers=1,
                 num_attention_heads=4,
                 intermediate_size=1500,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 out_aux_loss_weight=0.1,
                 out_dropout_prob=0.2,
                 out_num_attention_heads=6,
                 use_tl=True,
                 use_ha=True
                 ):
        super().__init__()
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
        # self.ic = torch.nn.Parameter(torch.zeros(1, seq_len, num_losses))
        self.ic_qa = torch.nn.Parameter(torch.zeros(1, seq_len, 2).cuda())
        if use_tl:
            self.ic_tl = torch.nn.Parameter(torch.zeros(1, seq_len, 1).cuda())
        else:
            self.ic_tl = torch.zeros(1, seq_len, 1).cuda()

        if use_ha:
            self.ic_ha = torch.nn.Parameter(torch.zeros(1, seq_len, 1).cuda())
        else:
            self.ic_ha = torch.zeros(1, seq_len, 1).cuda()

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
        ic = torch.cat((self.ic_qa, self.ic_tl, self.ic_ha), dim=-1)
        conditions = ic.expand((batch_size, -1, -1))
        for i in range(num_iters):
            hidden_states, conditions, loss = self.clayer(
                hidden_states, conditions, attention_mask,
                start_positions=start_positions, end_positions=end_positions,
                token_labels=token_labels, has_answers=has_answers
            )
            outputs.append(self.conditions2logits(conditions, batch_size))
            # losses.append(loss * self.loss_ratio**(self.num_iters-i-1))
            # losses.append(loss)
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
            # total_loss = sum(losses) / self.num_iters
            # total_loss = (losses[0] + losses[1] + losses[2]) / self.num_iters
            # total_loss = losses[-1]
            total_loss = loss
            # total_loss = losses
            # total_loss = torch.stack(losses).mean()
            # total_loss = losses[-1] + losses[-2]
            # total_loss = torch.nn.Parameter(torch.tensor(losses, requires_grad=True).mean())
            # total_loss = torch.tensor(losses, requires_grad=True).cuda().mean()
            # total_loss = losses[-1]
        else:
            total_loss = None

        # if start_positions is not None and end_positions is not None \
        #         and token_labels is not None and has_answers is not None:
            # total_loss = torch.tensor(losses, requires_grad=True).mean()
            # total_loss.append(losses)
        return outputs, total_loss


class Adaptor(nn.Module):
    def __init__(self, in_size, out_size, dropout_prob=0.2):
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
    def __init__(self,
                 enc_bert_model: str,
                 enc_selected_layers: List[int]=None,
                 # cir_num_iters: int=3,
                 cir_seq_len: int=448,
                 cir_num_losses: int=4,
                 cir_hidden_size: int=320,
                 cir_num_hidden_layers: int=1,
                 cir_num_attention_heads: int=4,
                 cir_intermediate_size: int=1280,
                 cir_hidden_dropout_prob: float=0.1,
                 cir_attention_probs_dropout_prob: float=0.1,
                 out_num_attention_heads: int=8,
                 out_aux_loss_weight: float=0.1,
                 out_dropout_prob: float=0.2,
                 use_tl=True,
                 use_ha=True
                 ):
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
        # self.cir_num_iters = cir_num_iters
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
            # num_iters=self.cir_num_iters,
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
        # logger.info("here!!!!: {}".format(x.size()))

        outputs, total_loss = self.cir(
            x, attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            token_labels=token_labels,
            has_answers=has_answers,
            num_iters=num_iters)

        # logger.info("here!!!!111: {}".format(x.size()))
        # total_loss = 0
        # if start_positions is not None and end_positions is not None \
        #         and token_labels is not None and has_answers is not None:
        #
        #     for i, rloss in enumerate(losses):
        #         sl, el, tl, hl = rloss
        #         for j, t in enumerate(tl):
        #             total_loss = \
        #                 total_loss + \
        #                 self.cir_loss_ratio**(self.cir_num_iters-1-i) * \
        #                 self.out_aux_loss_weight * t / conditions.size(0)
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * sl * 0.5
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * el * 0.5
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * self.out_aux_loss_weight * hl

        return outputs, total_loss


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
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
        """

        :param x: (B, S, H)
        :return:
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (B, S, A, HS)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (B, A, S, HS)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  # (B, S, H)
        # mixed_key_layer = self.key(hidden_states)      # (B, S, H)
        mixed_value_layer = self.value(hidden_states)  # (B, S, H)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (B, A, S, HS)
        # key_layer = self.transpose_for_scores(mixed_key_layer)      # (B, A, S, HS)
        key_layer = self.u                                          # (HS,)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (B, A, S, HS)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (B, A, S, HS) x (B, A, HS, S) -> (B, A, S, S)
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
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, S, A, HS)
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # (B, S, H)
        new_context_layer_shape = (context_layer.size(0), self.all_head_size)  # (B, H)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertQAL3Simplified(nn.Module):
    def __init__(self,
                 enc_bert_model: str,
                 enc_selected_layers: List[int]=None,
                 cir_num_iters: int=3,
                 cir_loss_ratio: float=0.5,
                 cir_num_losses: int=4,
                 cir_hidden_size: int=320,
                 cir_num_hidden_layers: int=1,
                 cir_num_attention_heads: int=4,
                 cir_intermediate_size: int=1280,
                 cir_hidden_dropout_prob: float=0.1,
                 cir_attention_probs_dropout_prob: float=0.1,
                 out_num_attention_heads: int=8,
                 out_aux_loss_weight: float=0.1,
                 out_dropout_prob: float=0.2
                 ):
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
        self.cir_num_iters = cir_num_iters
        self.cir_loss_ratio = cir_loss_ratio
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
        # self.cir = ConditionalRefinement(
        #     num_iters=self.cir_num_iters,
        #     loss_ratio=self.cir_loss_ratio,
        #     num_losses=self.cir_num_losses,
        #     hidden_size=self.cir_hidden_size,
        #     num_hidden_layers=self.cir_num_hidden_layers,
        #     num_attention_heads=self.cir_num_attention_heads,
        #     intermediate_size=self.cir_intermediate_size,
        #     hidden_dropout_prob=self.cir_hidden_dropout_prob,
        #     attention_probs_dropout_prob=self.cir_attention_probs_dropout_prob,
        #     out_aux_loss_weight=self.out_aux_loss_weight,
        #     out_dropout_prob=self.out_dropout_prob,
        #     out_num_attention_heads=self.out_num_attention_heads
        # )
        self.output = OutputLayerL3(
            self.cir_hidden_size, self.out_num_attention_heads, self.out_aux_loss_weight, self.out_dropout_prob)

    def forward(self, conditions, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                token_labels=None, has_answers=None):

        seq_out, _ = self.enc_bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)  # (B, S, H*L)
        x = torch.cat([seq_out[x] for x in self.enc_selected_layers], dim=-1)  # (B, S, H*ML)

        x = self.adaptor(x)  # (B, S, CH), CH: cir_hidden_size
        # logger.info("here!!!!: {}".format(x.size()))

        outputs, total_loss = self.output(
            x, attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            token_labels=token_labels,
            has_answers=has_answers)

        # outputs, total_loss = self.cir(
        #     x, conditions, attention_mask,
        #     start_positions=start_positions,
        #     end_positions=end_positions,
        #     token_labels=token_labels,
        #     has_answers=has_answers)

        # logger.info("here!!!!111: {}".format(x.size()))
        # total_loss = 0
        # if start_positions is not None and end_positions is not None \
        #         and token_labels is not None and has_answers is not None:
        #
        #     for i, rloss in enumerate(losses):
        #         sl, el, tl, hl = rloss
        #         for j, t in enumerate(tl):
        #             total_loss = \
        #                 total_loss + \
        #                 self.cir_loss_ratio**(self.cir_num_iters-1-i) * \
        #                 self.out_aux_loss_weight * t / conditions.size(0)
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * sl * 0.5
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * el * 0.5
        #         total_loss = total_loss + self.cir_loss_ratio**(self.cir_num_iters-1-i) * self.out_aux_loss_weight * hl

        return outputs, total_loss

