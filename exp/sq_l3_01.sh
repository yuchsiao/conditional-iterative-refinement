#!/usr/bin/env bash
python hb_train.py \
    --output_dir         save/l3_01 \
    --bert_model         bert-base-uncased\
    --do_lower_case \
    --train_file         data/train-v2.0.json \
    --dev_file           data/dev-v2.0.json \
    --dev_eval_file      data/dev_eval.json \
    --test_file          data/test-v2.0.json \
    --finetune \
    --selected_layers    -1 -5 -9 \
    --loss_weight_tl     0.1 \
    --loss_weight_ha     0.1 \
    --dropout_prob       0.1 \
    --metric_name        F1 \
    --eval_every         50000 \
    --max_seq_length     448 \
    --doc_stride         128 \
    --max_query_length   64 \
    --train_batch_size   32 \
    --learning_rate      5e-05 \
    --num_train_epochs   10 \
    --warmup_proportion  0.1 \
    --n_best_size        20 \
    --max_answer_length  30 \
    --seed               42 \
    --version_2_with_negative \
    --null_score_diff_threshold 0.0