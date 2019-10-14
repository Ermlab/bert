#!/usr/bin/env bash

# pierwszy parametr to nazwa datasetu czyli folder z backet np poleval wiki_content
# drugi parametr to nazwa fodleru z tfrecordsmi
# trzeci parametr to prefix do nazwy folderu z checkpointami
# czwarty parametr jest tylko w skryptach _eval i wskazuje on na dataset który bedzie testowany

./runner_1a.sh wiki_content tfrecords_seqlen_128_lower lower_lower
./runner_1b.sh wiki_content tfrecords_seqlen_128_lower lower_lower
./runner_2b.sh wiki_content tfrecords_seqlen_512_lower lower_lower
./runner_2b_eval.sh poleval tfrecords_test lower_lower wiki_content
