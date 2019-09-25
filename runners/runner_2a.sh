#!/usr/bin/env bash

 #Phase 2a

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/poleval/tfrecords/tfrecords_seqlen_512/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/poleval/checkpoints/test_v3_without_next_sentence_1a \
 --output_dir=gs://nlp-data-storage/poleval/checkpoints/test_v3_without_next_sentence_2a \
 --do_train=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --train_batch_size=1024 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=540000 \
 --num_warmup_steps=5000 \
 --learning_rate=5e-5 \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3 \
 --tpu-zone=us-central1-a \
 --save_checkpoints_steps=5000 \
 --iterations_per_loop=250 \
 --save_summary_steps=250
