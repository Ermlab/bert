#!/usr/bin/env bash

# Phase 1b

DATASET_NAME=$0
SEQ_LEN=$1
CHECKPOINT_PREFIX=$2

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/"$DATASET_NAME"/tfrecords/"$SEQ_LEN"/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/DATASET_NAME/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_1a/model.ckpt-270000 \
 --output_dir=gs://nlp-data-storage/DATASET_NAME/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_1b \
 --do_lower_case=False \
 --do_next_sentence_pred=False \
 --do_train=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --train_batch_size=1024 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=540000 \
 --num_warmup_steps=5000 \
 --learning_rate=5e-5 \
 --iterations_per_loop=250 \
 --save_summary_steps=250 \
 --save_checkpoints_steps=5000\
 --tpu_zone=us-central1-a \
 --use_tpu=True\
 --tpu_name=polish-nlp-tpu-1-v3
