#!/usr/bin/env bash
python hb_train.py \
    --output_dir         save/cir_320_it5_al03_ru2 \
    --do_lower_case \
    --train_file         data/train-v2.0.json \
    --dev_file           data/dev-v2.0.json \
    --dev_eval_file      data/dev_eval.json \
    --test_file          data/test-v2.0.json \
    \
    --finetune \
    \
    --bert_model                         bert-base-uncased \
    --enc_selected_layers                -1 -5 -9 \
    --cir_num_iters                      5 \
    --cir_num_losses                     4 \
    --cir_hidden_size                    320 \
    --cir_num_hidden_layers              1 \
    --cir_num_attention_heads            4 \
    --cir_intermediate_size              1280 \
    --cir_hidden_dropout_prob            0.2 \
    --cir_attention_probs_dropout_prob   0.2 \
    --out_num_attention_heads            8 \
    --out_aux_loss_weight                0.3 \
    --out_dropout_prob                   0.2 \
    \
    --metric_name        F1 \
    --eval_every         25000 \
    --max_seq_length     448 \
    --doc_stride         128 \
    --max_query_length   64 \
    --train_batch_size   32 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm      1.0 \
    --train_num_iters_increase_every_epoch  2.0 \
    --learning_rate      5e-05 \
    --num_train_epochs   10 \
    --warmup_proportion  0.1 \
    --n_best_size        20 \
    --max_answer_length  30 \
    --seed               42 \
    --version_2_with_negative \
    --null_score_diff_threshold 0.0


#    --cir_loss_ratio                     0.7 \
