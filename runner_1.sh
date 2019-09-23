#!/usr/bin/env bash

python3 run_pretraining.py \
 --input_file=gs://nl-data-storage/poleval/tfrecords/tfrecords/bert_dataset.tfrecords* \
 --output_dir=gs://nl-data-storage/poleval/test_v3_without_next_sentence\
 --do_train=True \
 --do_eval=True \
 --bert_config_file=gs://nl-data-storage/bert_config.json \
 --train_batch_size=1024 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=270000 \
 --num_warmup_steps=10000 \
 --learning_rate=1e-4 \
 --use_tpu=True \
 --tpu_name=bert\
 --save_checkpoints_steps=5000\
 --iterations_per_loop=250\
 --save_summary_steps=250
