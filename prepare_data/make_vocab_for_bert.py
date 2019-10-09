import argparse

import os
import sys
import sentencepiece as sp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument('input_path',
                        type=str,
                        help='Path to input')
    parser.add_argument('output_path',
                        type=str,
                        help='Path to output')
    parser.add_argument('-type',
                        '--model_type',
                        type=str,
                        choices=['bpe', 'unigram'],
                        default='unigram',
                        help='')
    parser.add_argument('-vsize',
                        '--vocab_size',
                        type=int,
                        default=30000,
                        help='Path to dataset folder with contexts .txt')
    parser.add_argument('-ssize',
                        '--input_sentence_size',
                        type=int,
                        help='Maximum number of sentences to process')
    parser.add_argument('-mask',
                        '--mask_special_tokens',
                        action='store_true',
                        help='Masks special tokens for specific one for bert eg.: <pad> for [PAD]')
    parser.add_argument('-s',
                        '--shuffle',
                        action='store_true',
                        help='shuffle input sentences')
    parser.add_argument('-name',
                        '--dataset_name',
                        type=str,
                        required=True,
                        help='Sources names, ordered alphabeticly, of dataset spaced with underscore. '
                             'eg.: clarin2000_wiki')

    args = parser.parse_args()

    data_file = args.input_path
    vocab_size = args.vocab_size
    model_type = args.model_type
    input_sentence_size = args.input_sentence_size
    dataset_name = args.dataset_name

    tok_model = os.path.join(
        args.output_path,
        f'{dataset_name}__{model_type}__voc_{vocab_size}__input_sentence_size_{input_sentence_size}'
    )

    cmd = f'--input={data_file}' \
          f' --model_prefix={tok_model}' \
          f' --model_type={model_type}' \
          f' --vocab_size={vocab_size}'
    if input_sentence_size:
        cmd += f' --input_sentence_size={input_sentence_size}'
    if args.mask_special_tokens:
        cmd += f' --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[SEP] --eos_piece=[CLS]' \
               f' --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3' \
               f' --user_defined_symbols=[MASK] '
    if args.shuffle:
        cmd += f' --shuffle_input_sentence=true'

    sp.SentencePieceTrainer.train(cmd)

    vocab_path = ".".join([tok_model, 'vocab'])
    our_vocab_path = "_".join([tok_model, 'parsed.vocab'])
    with open(vocab_path) as tok_vocab_file:
        with open(our_vocab_path, 'w+') as our_vocab_path:
            for line in tok_vocab_file:
                new_word = line.split('\t')[0]
                if new_word.startswith('[') and new_word.endswith(']\n'):
                    our_vocab_path.write(new_word)
                    continue
                if new_word.startswith('▁'):
                    new_word = new_word.replace('▁', '')
                else:
                    new_word = "".join(['##', new_word])
                our_vocab_path.write("".join([new_word, '\n']))

