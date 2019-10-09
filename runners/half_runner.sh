#!/usr/bin/env bash



./runner_1a.sh wiki_content tfrecords_seqlen_128_reversed_prefix_vocab lower_lower
./runner_1b.sh wiki_content tfrecords_seqlen_128_reversed_prefix_vocab lower_lower
./runner_2b.sh wiki_content tfrecords_seqlen_512_reversed_prefix_vocab lower_lower
