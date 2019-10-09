#!/usr/bin/env bash

DATASET_NAME=$0
SEQ_LEN=$1
CHECKPOINT_PREFIX=$2

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/"$DATASET_NAME"/tfrecords/"$SEQ_LEN"/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/"$DATASET_NAME"/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_1a/model.ckpt-540000 \
 --output_dir=gs://nlp-data-storage/"$DATASET_NAME"/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_1b_eval \
 --do_next_sentence_pred=True \
 --do_train=False \
 --do_eval=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --eval_batch_size=128 \
 --max_seq_length=128 \
 --iterations_per_loop=128\
 --max_eval_steps=128 \
 --save_summary_steps=128 \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3 \
 --tpu_zone=us-central1-a
