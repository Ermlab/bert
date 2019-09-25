#!/usr/bin/env bash

#Phase 2a eval

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/poleval/tfrecords/tfrecords_test_seqlen_128/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/poleval/checkpoints/test_v3_without_next_sentence_2a/ \
 --output_dir=gs://nlp-data-storage/poleval/checkpoints/test_v3_without_next_sentence_2a_eval \
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
 --tpu-zone=us-central1-a
