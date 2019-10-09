#!/usr/bin/env bash



./runner_1a.sh wiki_content tfrecords_seqlen_128 lower_lower
./runner_1b.sh wiki_content tfrecords_seqlen_128 lower_lower
./runner_2b.sh wiki_content tfrecords_seqlen_512 lower_lower
./runner_2b_eval.sh poleval tfrecords_test lower_lower
