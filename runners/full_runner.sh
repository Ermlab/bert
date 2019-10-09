#!/usr/bin/env bash



./runner_1a.sh wiki_content tfrecords_seqlen_128 uper_uper
./runner_1a_eval.sh poleval tfrecords_test uper_uper wiki_content
./runner_2a.sh wiki_content tfrecords_seqlen_512 uper_uper
./runner_2a_eval.sh poleval tfrecords_test uper_uper wiki_content
./runner_1b.sh wiki_content tfrecords_seqlen_128 uper_uper
./runner_1b_eval.sh poleval tfrecords_test uper_uper wiki_content
./runner_2b.sh wiki_content tfrecords_seqlen_512 uper_uper
./runner_2b_eval.sh poleval tfrecords_test uper_uper wiki_content
