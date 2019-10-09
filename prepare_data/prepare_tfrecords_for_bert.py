import os
import argparse
import sys
import tempfile

from typing import TextIO

CMD = """
python {path_to_bert} \
  --input_file={input_file} \
  --output_file={output_file} \
  --vocab_file={vocab_file} \
  --do_lower_case=False \
  --do_whole_word_mask=True \
  --max_seq_length={bptt} \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15
  """


def extract_article_by_blank_lines(txt_file: TextIO) -> str:
    """
    Reads text file and splits it by blank lines

    Parameters
    ----------
    txt_file: TextIO
        opened text file

    Yields
    -------
    str:
        string with joined lines
    """
    lines = ''
    for line in txt_file:
        if line == "\n" and lines != '':
            yield lines
            lines = ''
        elif line != '\n':
            lines = "".join([lines, line])
    yield lines


def extract_article_by_number_of_sentence(txt_file: TextIO, split_after: int) -> str:
    """
    Reads text file and splits it by number of lines

    Parameters
    ----------
    txt_file: TextIO
        opened text file

    Yields
    -------
    str:
        string with joined lines
    """
    lines = ''
    for i, line in enumerate(txt_file):
        if i % split_after == 0:
            yield lines
            lines = ''
        else:
            lines = "".join([lines, line])
    yield lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument('corpus_path',
                        type=str,
                        help='Path to corpus .txt file')
    parser.add_argument('vocab_path',
                        type=str,
                        help='Path to vocab file')
    parser.add_argument('output_path',
                        type=str,
                        help='Path to folder where tfrecords will be saved')
    parser.add_argument('bert_path',
                        type=str,
                        help='Path to folder where is bert repository')
    parser.add_argument('-sl',
                        '--split_by_lines',
                        type=int,
                        help='Changes default corpus splitting by blank line to by line number specify by this parameter.'
                             ' Leave blank to use splitting by blank line')
    parser.add_argument('-tmp_dir_path',
                        '--temporary_directory_path',
                        type=str,
                        default='./',
                        help='Path to folder where temporary files will be saved')
    parser.add_argument('--bptt',
                        type=int,
                        default=128,
                        help='Max sequence length')
    parser.add_argument('-cp',
                        '--checkpoint',
                        type=int,
                        default=0,
                        help='Run from specific iteration number')
    args = parser.parse_args()

    EXTRACTION_FACTORY = {
        'blank': extract_article_by_blank_lines,
        'lines': extract_article_by_number_of_sentence,
    }

    corpus_path = os.path.abspath(args.corpus_path)
    bert_path = os.path.abspath(args.bert_path)
    vocab_path = os.path.abspath(args.vocab_path)
    output_path = os.path.abspath(args.output_path)
    temporary_directory_path = os.path.abspath(args.temporary_directory_path)
    checkpoint = args.checkpoint

    with open(corpus_path, 'r') as corpus_file, \
            tempfile.TemporaryDirectory(dir=temporary_directory_path) as tmp_file_dir:
        if args.split_by_lines:
            print('Splitting by number of lines')
            gen = extract_article_by_number_of_sentence(corpus_file, args.split_by_lines)
        else:
            print('Splitting by blank lines')
            gen = extract_article_by_blank_lines(corpus_file)

        for i, article in enumerate(gen):
            if i >= checkpoint:
                with tempfile.NamedTemporaryFile(mode='w+', dir=tmp_file_dir) as tmp_file:
                    tmp_file.write(article)
                    tmp_file.seek(0)

                    os.system(CMD.format(
                        input_file=tmp_file.name,
                        output_file=os.path.join(output_path, f"bert_dataset.tfrecords{i}"),
                        vocab_file=vocab_path,
                        bptt=args.bptt,
                        path_to_bert=bert_path
                    ))

