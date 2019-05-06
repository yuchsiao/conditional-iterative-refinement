
from __future__ import absolute_import, division, print_function

import argparse
import collections
import datetime
import json
import logging
import math
import os
import random
import sys
from typing import Dict, List, Tuple
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
import util
from hb_util import \
    read_squad_examples,\
    convert_examples_to_features,\
    RawResult,\
    extract_predictions,\
    write_predictions,\
    CheckpointSaver,\
    InputFeatures
from hb_metric import extract_eval_answers, eval_dicts_uuid
from hb_model import \
    BertQAL3
    # BertQAL3Simplified as BertQAL3
import pickle

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                     "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                     "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    ## Ops
    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.", default=True)
    # parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.", default=True)
    parser.add_argument("--eval_every", type=int, help="Whether to run eval on the dev set.", default=10000)

    ## Data sources
    parser.add_argument("--train_file", default="data/train-v2.0.json",
                        type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file", default="data/dev-v2.0.json", type=str,
                        help="SQuAD json for dev. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--dev_eval_file", default="data/dev_eval.json", type=str,
                        help="SQuAD eval for predictions")
    parser.add_argument("--test_file", default="data/test-v2.0.json", type=str,
                        help="SQuAD json for test. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    ## Token parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    ## Model parameters
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--enc_selected_layers", type=int, nargs="+", default=[-1, -5, -9])
    parser.add_argument("--cir_num_iters", type=int, default=3)
    parser.add_argument("--cir_loss_ratio", type=float, default=0.7)
    parser.add_argument("--cir_num_losses", type=int, default=4)
    parser.add_argument("--cir_hidden_size", type=int, default=320)
    parser.add_argument("--cir_num_hidden_layers", type=int, default=1)
    parser.add_argument("--cir_num_attention_heads", type=int, default=4)
    parser.add_argument("--cir_intermediate_size", type=int, default=1280)
    parser.add_argument("--cir_hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--cir_attention_probs_dropout_prob", type=float, default=0.2)
    parser.add_argument("--out_num_attention_heads", type=int, default=8)
    parser.add_argument("--out_aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--out_dropout_prob", type=float, default=0.2)
    parser.add_argument("--use_tl", type=str2bool, nargs='?', const=True, default='yes')
    parser.add_argument("--use_ha", type=str2bool, nargs='?', const=True, default='yes')

    ## Training parameters
    parser.add_argument("--train_num_iters_inc_every_global_step", default=200, type=int)
    parser.add_argument("--train_num_iters_inc_randomness", default=0.2, type=float)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="BertAdam grad clipping")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--local_rank",
    #                     type=int,
    #                     default=-1,
    #                     help="local_rank for distributed training on gpus")
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--finetune',
                        action='store_true',
                        help='Finetune the whole model')
    parser.add_argument('--tunek', type=int, default=2,
                        help='Relax the last k transformer layers')
    parser.add_argument('--cache_prerun', action='store_true',
                        help="cache prerun results")

    args = parser.parse_args()
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'
                         .format(args.metric_name))
    return args


args = parse_args()

# Create output_dir
if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    print("WARNING: Output directory () already exists and is not empty.")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

NAME = 'train'
# ROOT = '.'
PROCESS_START_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

formatter = logging.Formatter('%(asctime)s %(name)12s [%(levelname)s]\t%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler(os.path.join(args.output_dir, "{}_{}.log".format(NAME, PROCESS_START_TIME)))
fh.setFormatter(formatter)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(ch)
logging.getLogger().addHandler(fh)

logger = logging.getLogger(__name__)

#
# def verify_impossible(features: List[InputFeatures]) -> None:
#     for f in features:
#         assert (f.is_impossible and f.start_position == 0 and f.end_position == 0) or \
#                (not f.is_impossible and f.start_position > 0 and f.end_position > 0), \
#             "inconsistent features: {}".format(f.__dict__)
#


def generate_token_labels(features: List[InputFeatures]) -> None:
    """Genereate token-level binary labels for each feature

    :param features: output of convert_examples_to_features
    :return: features with token_labels
    """
    for f in features:
        f.token_labels = [0] * len(f.input_mask)
        for i in range(f.start_position, f.end_position+1):
            f.token_labels[i] = 1


def generate_impossible_labels(features: List[InputFeatures]) -> None:
    for f in features:
        if f.start_position == 0 and f.end_position == 0:
            f.is_impossible = True
        elif f.start_position > 0 and f.end_position > 0:
            f.is_impossible = False
        else:
            raise ValueError("data issue: start_position: {}, end_position: {}".format(
                f.start_position, f.end_position))


def load_examples_to_features(file, tokenizer, args, name=None):
    train_examples = read_squad_examples(input_file=file)

    cached_train_features_file = file + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length))
    train_features = None
    try:
        # use cache
        with open(cached_train_features_file, "rb") as reader:
            logger.info("{} uses cache".format(name))
            train_features = pickle.load(reader)
    except:
        # no cache. rebuild it.
        logger.info("{} has no cache, building it".format(name))
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(train_features, writer)

    logger.info("Generate additional labels...")
    generate_token_labels(train_features)
    generate_impossible_labels(train_features)
    logger.info("Generate additional labels... DONE")

    logger.info("***** {} Data *****".format(name))
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    # all_example_idxs = torch.tensor([f.example_index for f in train_features], dtype=torch.long)
    # all_example_index = torch.tensor([f.example_index for f in train_features], dtype=torch.long)
    all_index = torch.tensor([f.index for f in train_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    token_labels = torch.tensor([f.token_labels for f in train_features], dtype=torch.long)
    has_answers = torch.tensor([not f.is_impossible for f in train_features], dtype=torch.long)

    # print([f.start_position for f in train_features])
    all_start_positions = \
        torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = \
        torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    return train_examples, train_features, all_example_index, \
           all_index, all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions,\
           token_labels, has_answers


def predict_dataset(model, dataset_name, examples, features, dataset, global_step, device, n_gpu,
                    all_prerun=None, golds=None, num_losses=4, num_iters=1):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size * args.gradient_accumulation_steps)
    nll_meter = util.AverageMeter()

    def extract_raw_results_from_output(outputs, example_indices, features, res):
        for iter, output in enumerate(outputs):
            batch_start_logits, batch_end_logits, batch_tl_logits, batch_ha_logits = output
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                tl_logits = batch_tl_logits[i].detach().cpu().tolist()
                ha_logits = batch_ha_logits[i].detach().cpu().tolist()
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                res[iter].append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits,
                        tl_logits=tl_logits,
                        ha_logits=ha_logits))

    res = [list() for _ in range(num_iters)]
    for batch in tqdm(dataloader, desc="Iteration"):
        # if n_gpu == 1:
        batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        row_index, input_ids, input_mask, segment_ids, example_indices, \
            start_positions, end_positions, tls, has_ans = batch
        # batch_size = input_ids.size(0)
        # flag_size1 = False
        # if batch_size == 1:
        #     flag_size1 = True
        #     ex_num = 4
        #     row_index = row_index.expand([ex_num] + [-1] * (len(row_index.size()) - 1))
        #     input_ids = input_ids.expand([ex_num] + [-1] * (len(input_ids.size()) - 1))
        #     segment_ids = segment_ids.expand([ex_num] + [-1] * (len(segment_ids.size()) - 1))
        #     start_positions = start_positions.expand([ex_num] + [-1] * (len(start_positions.size()) - 1))
        #     end_positions = end_positions.expand([ex_num] + [-1] * (len(end_positions.size()) - 1))
        #     tls = tls.expand([ex_num] + [-1] * (len(tls.size()) - 1))
        #     has_ans = has_ans.expand([ex_num] + [-1] * (len(has_ans.size()) - 1))

        # Prepare init conditions
        # conditions = torch.zeros(input_ids.size(0), input_ids.size(1), num_losses).to(device)

        if args.finetune:
            with torch.no_grad():
                try:
                    outputs, total_loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, tls,
                                                has_ans, num_iters=num_iters)
                except TypeError as e:
                    logger.warning("Pred input_ids errors. Skip it")
                    continue
                # if flag_size1:
                #     for i in range(len(outputs)):
                #         batch_start_logits, batch_end_logits, tl_logits, ha_logits = outputs[i]
                #         batch_start_logits = batch_start_logits[0].unsqueeze(0)
                #         batch_end_logits = batch_end_logits[0].unsqueeze(0)
                #         ha_logits = ha_logits[0].unsqueeze(0)
                #         outputs[i] = (batch_start_logits, batch_start_logits, batch_end_logits, ha_logits)

                # batch_start_logits, batch_end_logits = outputs[-1]
                # batch_start_logits, batch_end_logits, tl_logits, ha_logits = outputs[-1]
                # batch_start_logits, batch_end_logits, tl_logits, ha_logits = outputs
        else:
            prerun = all_prerun[row_index].to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, total_loss = model(
                    prerun, segment_ids, input_mask, start_positions, end_positions, tls, has_ans
                )
        if n_gpu > 1:
            total_loss = total_loss.mean()
        nll_meter.update(total_loss, start_positions.size(0))

        # extract raw results for every output
        extract_raw_results_from_output(outputs, example_indices, features, res)

        # for i, example_index in enumerate(example_indices):
        #     start_logits = batch_start_logits[i].detach().cpu().tolist()
        #     end_logits = batch_end_logits[i].detach().cpu().tolist()
        #     eval_feature = features[example_index.item()]
        #     unique_id = int(eval_feature.unique_id)
        #     res.append(RawResult(unique_id=unique_id,
        #                                  start_logits=start_logits,
        #                                  end_logits=end_logits))

    all_preds = []
    all_nbest = []
    all_scores_diff = []
    for r in res:
        preds, all_nbest_json, scores_diff_json = \
            extract_predictions(examples, features, r,
                                args.n_best_size, args.max_answer_length,
                                args.do_lower_case, args.verbose_logging,
                                args.version_2_with_negative, args.null_score_diff_threshold)
        all_preds.append(preds)
        all_nbest.append(all_nbest_json)
        all_scores_diff.append(scores_diff_json)

    from sklearn.metrics import roc_auc_score

    def eval_tls(features, res):
        assert len(features) == len(res), "not the same lengths {} vs {}".format(len(features), len(res))
        tls = np.array([f.token_labels for f in features], dtype=np.float32)
        masks = np.array([f.segment_ids for f in features], dtype=np.float32)
        aucs = 0
        for i in range(tls.shape[0]):
            with torch.no_grad():
                ps = torch.tensor(res[i].tl_logits)
                ms = torch.tensor(masks[i])
                ms[0] = 1
                # print("ps: {}".format(ps.size()))
                # print("ms: {}".format(ms.size()))
                # print("ps: {}".format(ps))
                # print("ms: {}".format(ms))
                preds = torch.nn.functional.softmax(ps + (1.0 - ms) * -10000.0, dim=0)
                # print("compute auc: {}".format(i))
                # if features[i].is_impossible:
                #     print("preds: {}".format(preds))
                #     print("tls: {}".format(tls[i]))
                #     print("mask: {}".format(ms))
                aucs += roc_auc_score(tls[i], preds.numpy(), sample_weight=ms)
        avg_auc = aucs / len(features)
        return avg_auc * 100

    def eval_has(features, res):
        assert len(features) == len(res), "not the same lengths {} vs {}".format(len(features), len(res))
        has = np.array([not f.is_impossible for f in features], dtype=np.float32)
        preds = np.array([x.ha_logits for x in res])
        auc = roc_auc_score(has, preds)
        return auc * 100

    all_evals = None
    if golds:
        all_evals = []
        for preds in all_preds:
            evals = eval_dicts_uuid(golds, preds)
            evals["NLL"] = nll_meter.avg
            all_evals.append(evals)

        for iter, r in enumerate(res):
            all_evals[iter]["tl_avg_auc"] = eval_tls(features, r)
            all_evals[iter]["ha_auc"] = eval_has(features, r)

    for i in range(num_iters):
        output_prediction_file = os.path.join(args.output_dir, "{}_predictions_{}_iter{}.json".format(dataset_name, global_step, i+1))
        output_nbest_file = os.path.join(args.output_dir, "{}_nbest_predictions_{}_iter{}.json".format(dataset_name, global_step, i+1))
        output_null_log_odds_file = os.path.join(args.output_dir, "{}_null_odds_{}_iter{}.json".format(dataset_name, global_step, i+1))
        write_predictions(
            all_preds[i], all_nbest[i], all_scores_diff[i],
            output_prediction_file, output_nbest_file, output_null_log_odds_file)

    return all_evals


def main():
    logger.info("{}".format(json.dumps(vars(args), indent=4)))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare model
    model = BertQAL3(
        enc_bert_model=args.bert_model,
        enc_selected_layers=args.enc_selected_layers,
        # cir_num_iters=args.cir_num_iters,
        cir_seq_len=args.max_seq_length,
        cir_num_losses=args.cir_num_losses,
        cir_hidden_size=args.cir_hidden_size,
        cir_num_hidden_layers=args.cir_num_hidden_layers,
        cir_num_attention_heads=args.cir_num_attention_heads,
        cir_intermediate_size=args.cir_intermediate_size,
        cir_hidden_dropout_prob=args.cir_hidden_dropout_prob,
        cir_attention_probs_dropout_prob=args.cir_attention_probs_dropout_prob,
        out_num_attention_heads=args.out_num_attention_heads,
        out_aux_loss_weight=args.out_aux_loss_weight,
        out_dropout_prob=args.out_dropout_prob,
        use_tl=args.use_tl,
        use_ha=args.use_ha
    )
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load data
    #   Train set
    train_examples, train_features, _, \
        all_index, all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions,\
        token_labels, has_answers =\
        load_examples_to_features(args.train_file, tokenizer, args, name="train")
    #   Dev set
    dev_examples, dev_features, dev_all_example_index, \
        dev_all_index, dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, \
        dev_all_start_positions, dev_all_end_positions,\
        dev_token_labels, dev_has_answers =\
        load_examples_to_features(args.dev_file, tokenizer, args, name="dev")
    with open(args.dev_eval_file) as fin:
        golds = json.load(fin)
    golds = extract_eval_answers(golds)
    #   Test set
    if args.predict_test:
        test_examples, test_features, test_all_example_index, \
            test_all_index, test_all_input_ids, test_all_input_mask, test_all_segment_ids, \
            test_all_start_positions, test_all_end_positions,\
            test_token_labels, test_has_answers =\
            load_examples_to_features(args.test_file, tokenizer, args, name="test")

    # Prepare optimizer
    num_train_optimization_steps = int(
        all_index.size(0) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]  # remove pooler, which is not used
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps,
                         max_grad_norm=args.max_grad_norm)

    train_data = TensorDataset(
        all_index, all_input_ids, all_input_mask, all_segment_ids,
        all_start_positions, all_end_positions, token_labels, has_answers)

    dev_data = TensorDataset(
        dev_all_index, dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids,
        dev_all_example_index,
        dev_all_start_positions, dev_all_end_positions, dev_token_labels, dev_has_answers)

    if args.predict_test:
        test_data = TensorDataset(
            test_all_index, test_all_input_ids, test_all_input_mask, test_all_segment_ids,
            test_all_example_index,
            test_all_start_positions, test_all_end_positions, test_token_labels, test_has_answers)

    all_prerun = None
    dev_all_prerun = None
    test_all_prerun = None
    if not args.finetune:
        # Prepare BERT train prerun
        prerun_sampler = SequentialSampler(train_data)
        prerun_dataloader = DataLoader(train_data, sampler=prerun_sampler, batch_size=args.train_batch_size)
        logger.info("Start BERT train prerun...")
        cached_train_prerun = '{}_{}_{}_prerun.npy'.format(
            args.train_file, list(filter(None, args.bert_model.split('/'))).pop(), args.tunek)
        try:
            # use cache
            logger.info("  Loading train prerun into cached file %s", cached_train_prerun)
            all_prerun = torch.from_numpy(np.load(cached_train_prerun))
        except:
            # no cache. rebuild it.
            logger.info("  No train prerun cached. Building it")
            prerun_res = []
            model.eval()
            for step, batch in enumerate(tqdm(prerun_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                _, input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                sequence_output = model(input_ids, segment_ids, input_mask, prerun=True)
                prerun_res.append(sequence_output.cpu().detach())
            all_prerun = torch.cat(prerun_res, dim=0)
            logger.info("  Saving train prerun into cached file %s", cached_train_prerun)
            if args.cache_prerun:
                np.save(cached_train_prerun, all_prerun.numpy(), allow_pickle=False)
        logger.info("Start BERT train prerun... DONE")

        # Prepare BERT dev prerun
        prerun_sampler = SequentialSampler(dev_data)
        prerun_dataloader = DataLoader(dev_data, sampler=prerun_sampler, batch_size=args.train_batch_size)
        logger.info("Start BERT dev prerun...")
        cached_dev_prerun = '{}_{}_{}_prerun.npy'.format(
            args.dev_file, list(filter(None, args.bert_model.split('/'))).pop(), args.tunek)
        try:
            # use cache
            logger.info("  Loading dev prerun into cached file %s", cached_dev_prerun)
            dev_all_prerun = torch.from_numpy(np.load(cached_dev_prerun))
        except:
            logger.info("  No dev prerun cached. Building it")
            prerun_res = []
            model.eval()
            for step, batch in enumerate(tqdm(prerun_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                _, input_ids, input_mask, segment_ids, _, _, _ = batch
                sequence_output = model(input_ids, segment_ids, input_mask, prerun=True)
                prerun_res.append(sequence_output.cpu().detach())
            dev_all_prerun = torch.cat(prerun_res, dim=0)
            logger.info("  Saving dev prerun into cached file %s", cached_dev_prerun)
            if args.cache_prerun:
                np.save(cached_dev_prerun, dev_all_prerun.numpy(), allow_pickle=False)
        logger.info("Start BERT dev prerun... DONE")

        # Prepare BERT test prerun
        prerun_sampler = SequentialSampler(test_data)
        prerun_dataloader = DataLoader(test_data, sampler=prerun_sampler, batch_size=args.train_batch_size)
        logger.info("Start BERT test prerun...")
        cached_test_prerun = '{}_{}_{}_prerun.npy'.format(
            args.test_file, list(filter(None, args.bert_model.split('/'))).pop(), args.tunek)
        try:
            # use cache
            logger.info("  Loading test prerun into cached file %s", cached_test_prerun)
            test_all_prerun = torch.from_numpy(np.load(cached_test_prerun))
        except:
            logger.info("  No test prerun cached. Building it")
            prerun_res = []
            model.eval()
            for step, batch in enumerate(tqdm(prerun_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                _, input_ids, input_mask, segment_ids, _, _, _ = batch
                sequence_output = model(input_ids, segment_ids, input_mask, prerun=True)
                prerun_res.append(sequence_output.cpu().detach())
            test_all_prerun = torch.cat(prerun_res, dim=0)
            logger.info("  Saving test prerun into cached file %s", cached_test_prerun)
            if args.cache_prerun:
                np.save(cached_test_prerun, test_all_prerun.numpy(), allow_pickle=False)
        logger.info("Start BERT test prerun... DONE")


    # Training
    global_step = 0
    report_counter = 0
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    train_nll_meter = util.AverageMeter()
    saver = CheckpointSaver(args.output_dir,
                            max_checkpoints=args.max_checkpoints,
                            metric_name=args.metric_name,
                            maximize_metric=args.maximize_metric,
                            log=logger)
    model.train()
    # seq_len = args.max_seq_length
    last_num_iters_wo_randomness = 0
    for ei in trange(int(args.num_train_epochs), desc="Epoch"):
        # num_iters = min(args.cir_num_iters, round(1 + ei * args.train_num_iters_increase_every_epoch))
        # logger.info("### Epoch={}, num_iters={}".format(ei+1, num_iters))
        logger.info("### Epoch={}".format(ei+1))

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            num_iters = min(args.cir_num_iters, (global_step // args.train_num_iters_inc_every_global_step) + 1)
            if num_iters > last_num_iters_wo_randomness:
                logger.info("##### num_iters increased to {} @ {}".format(num_iters, global_step))
                last_num_iters_wo_randomness = num_iters
            if random.random() < args.train_num_iters_inc_randomness:
                num_iters += +1 if random.random() > 0.5 else -1
                num_iters = max(num_iters, 1)
            # Prepare init conditions
            # conditions = torch.zeros(batch[0].size(0), seq_len, args.cir_num_losses)
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                # conditions = conditions.to(device)
            row_index, input_ids, input_mask, segment_ids, start_positions, end_positions, tls, has_ans = batch
            batch_size = input_ids.size(0)
            if batch_size < args.train_batch_size:
                continue
            # flag_size1 = False
            # if batch_size == 1:
            #     flag_size1 = True
            #     ex_num = 4
            #     row_index = row_index.expand([ex_num] + [-1] * (len(row_index.size()) - 1))
            #     input_ids = input_ids.expand([ex_num] + [-1] * (len(input_ids.size()) - 1))
            #     segment_ids = segment_ids.expand([ex_num] + [-1] * (len(segment_ids.size()) - 1))
            #     start_positions = start_positions.expand([ex_num] + [-1] * (len(start_positions.size()) - 1))
            #     end_positions = end_positions.expand([ex_num] + [-1] * (len(end_positions.size()) - 1))
            #     tls = tls.expand([ex_num] + [-1] * (len(tls.size()) - 1))
            #     has_ans = has_ans.expand([ex_num] + [-1] * (len(has_ans.size()) - 1))
            # if step == 8185:
            #     logger.info("input_ids @ {}: {}".format(step, input_ids.size()))
            #     logger.info("input_ids @ {}: {}".format(step, input_ids))

            # Forward
            if args.finetune:
                try:
                    __output_logits, loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, tls, has_ans, num_iters=num_iters)
                except TypeError as e:
                    logger.warning("Train input_ids errors. Skip it")
                    continue
            # else:
            #     prerun = all_prerun[row_index].to(device)
            #     _, _, loss = model(prerun, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                # logger.info("loss: {}".format(loss))
            train_nll_meter.update(loss.item(), args.train_batch_size)

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            report_counter += args.train_batch_size
            if report_counter >= args.eval_every:
                report_counter -= args.eval_every
                # logger.info("Evaluating at {} samples".format(global_step * args.train_batch_size * args.gradient_accumulation_steps))
                # logger.info("gs: {}, batch_size: {}".format(global_step, args.train_batch_size))
                model.eval()
                # Evaluate on dev set
                # logger.info("Evaluating on dev set...")
                all_evals = predict_dataset(
                    model, "dev", dev_examples, dev_features, dev_data, global_step, device, n_gpu,
                    all_prerun=dev_all_prerun, golds=golds, num_losses=args.cir_num_losses,
                    num_iters=args.cir_num_iters)

                for i, evals in enumerate(all_evals, start=1):
                    evals["TrNLL"] = train_nll_meter.avg
                    logger.info("DEV@iter{}: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} @{},{}".format(
                        i,
                        evals["EM"], evals["F1"], evals["AvNA"], evals["NLL"], evals["TrNLL"],
                        evals["tl_avg_auc"], evals["ha_auc"],
                        global_step * args.train_batch_size * args.gradient_accumulation_steps,
                        global_step
                    ))

                # If best, evaluate on test set
                if args.predict_test and saver.is_best(all_evals[-1][args.metric_name]):
                    logger.info("Best so far. Predicting test set...")
                    predict_dataset(
                        model, "test", test_examples, test_features, test_data, global_step, device, n_gpu,
                        all_prerun=test_all_prerun, golds=None, num_losses=args.cir_num_losses,
                        num_iters=args.cir_num_iters)

                # Save model
                saver.save(global_step, model, all_evals[-1][args.metric_name], device)
                train_nll_meter.reset()
                model.train()

    # Save a trained model and the associated configuration
    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    # torch.save(model_to_save.state_dict(), output_model_file)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    # with open(output_config_file, 'w') as f:
    #     f.write(model_to_save.config.to_json_string())


if __name__ == "__main__":
    main()
