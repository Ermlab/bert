#!/usr/bin/env bash


#Phase 2b

python3 ../run_pretraining.py \
 --input_file=gs://nlp-data-storage/poleval/tfrecords/tfrecords_seqlen_512/bert_dataset.tfrecords* \
 --init_checkpoint=gs://nlp-data-storage/poleval/checkpoints/without_next_sentence_1b/model.ckpt-270000 \
 --output_dir=gs://nlp-data-storage/poleval/checkpoints/without_next_sentence_2b \
 --do_next_sentence_pred=False \
 --do_train=True \
 --bert_config_file=gs://nlp-data-storage/bert_config.json \
 --train_batch_size=128 \
 --max_seq_length=512 \
 --max_predictions_per_seq=20 \
 --num_train_steps=30000 \
 --num_warmup_steps=10000 \
 --learning_rate=5e-5 \
 --tpu_zone=us-central1-a \
 --use_tpu=True \
 --tpu_name=polish-nlp-tpu-1-v3 \
 --save_checkpoints_steps=5000 \
 --iterations_per_loop=250 \
 --save_summary_steps=250
