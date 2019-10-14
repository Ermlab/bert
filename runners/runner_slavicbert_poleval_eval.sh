#!/usr/bin/env bash

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/poleval/tfrecords/tfrecords_test/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/slavic_bert/checkpoints/bert_model.ckpt \
 --output_dir=gs://nlp-data-storage/slavic_bert/eval \
 --do_next_sentence_pred=True \
 --do_train=False \
 --do_eval=True \
 --bert_config_file=gs://nlp-data-storage/slavic_bert/config.json \
 --eval_batch_size=128 \
 --max_seq_length=128 \
 --iterations_per_loop=128\
 --max_eval_steps=128 \
 --save_summary_steps=128 \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3 \
 --tpu_zone=us-central1-a
