#!/usr/bin/env bash

#Phase 1a

DATASET_NAME=$1
SEQ_LEN=$2
CHECKPOINT_PREFIX=$3

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/"$DATASET_NAME"/tfrecords/"$SEQ_LEN"/bert_dataset.tfrecords* \
 --output_dir=gs://nlp-data-storage/"$DATASET_NAME"/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_1a \
 --do_next_sentence_pred=False \
 --do_train=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --train_batch_size=1024 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=270000 \
 --num_warmup_steps=10000 \
 --learning_rate=1e-4 \
 --save_checkpoints_steps=5000\
 --iterations_per_loop=250 \
 --save_summary_steps=250 \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3 \
 --tpu_zone=us-central1-a






