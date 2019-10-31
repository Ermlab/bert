# Creating dataset for bert-tf
Pipeline for creating dataset for training bert-tf from google-research 

## Make corpus (one .txt file)
Corpus needs to be a specially parsed text file. 
It needs to have one sentence per line. Each sentence needs to be tokenized (nltk word tokenizer). Each one document/article needs one empty space before seconds one.
### Usage
```bash
python -m playground.bert.make_corpus_for_bert -wiki <WIKIPEDIA_PATH> -ds <DATASET_PATH> output_path
```

## Make vocab (sentencepiece)
Make vocabulary using sentencepiece from google research. 
For bert-tf use parameter --mask_special_tokens to mask default special tokens like <pad> to [PAD] and to add [MASK] token.


### Usage
```bash
python -m playground.bert.make_vocab_for_bert <INPUT_PATH> <OUTPUT_PATH>\
    --model_type <BPE,UNIGRAM> \
    --vocab_size <VOCAB_SIZE> \
    --input_sentence_size <INPUT_SENTENCE_SIZE> \
    --dataset_name <DATASET_NAME> \     
    --mask_special_tokens \
    --shuffle 
```

## Prepare batched .tfrecords
Runs script from bert-tf create_pretraining_data.py in batch. Script needs to be run in batches because it needs lots of RAM.
Script iterate through corpus file and saves one article (One set of sentences ended with blank line [\n]) into temporary file then runs create_pretraining_data.py script.      

### Usage
```bash
python -m playgroprepare_tfrecords_for_bert.py <CORPUS_PATH> <VOCAB_PATH> <OUTPUT_PATH> \ 
    -tmp_dir_path <TEMPORARY_DIRECTORY_PATH> \
    --bptt <BPTT>
```

## Training
### Runners
For training u can use any runner in ./bert/runners

for training runners you must provide the following environment variables:
- DATASET_NAME - name folder where tfrecords are placed and where checkpoints folder is located
- SEQ_LEN - sequence length
- CHECKPOINT_PREFIX - prefix of folder where checkpoint will be created

for evaluation runners you must provide the following environment:
- DATASET_NAME - name folder where testing tfrecords are placed 
- SEQ_LEN - sequence length
- CHECKPOINT_PREFIX - prefix of folder where checkpoint is located
- CHECKPOINT_FOLDER - name of folder where checkpoint is located (training dataset name)
