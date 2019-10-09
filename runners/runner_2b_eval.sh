#!/usr/bin/env bash

#Phase 2a eval

DATASET_NAME=$1
SEQ_LEN=$2
CHECKPOINT_PREFIX=$3
CHECKPOINT_FOLDER=$4

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/"$DATASET_NAME"/tfrecords/"$SEQ_LEN"/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/"$CHECKPOINT_FOLDER"/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_2b/ \
 --output_dir=gs://nlp-data-storage/"$CHECKPOINT_FOLDER"/checkpoints/"$CHECKPOINT_PREFIX"_without_next_sentence_2b_eval \
 --do_next_sentence_pred=True \
 --do_train=False \
 --do_eval=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --eval_batch_size=128 \
 --max_seq_length=128 \
 --iterations_per_loop=128\
 --max_eval_steps=128 \
 --save_summary_steps=128 \
 --tpu_zone=us-central1-a \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3
