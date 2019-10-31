from __future__ import annotations

import csv
import os
import sys
import argparse
from typing import Generator, Iterable, TextIO

from nltk.tokenize import sent_tokenize, word_tokenize


def get_tokenized_text(data: Iterable, row_label: str) -> Generator[str, None, None]:
    for row in data:
        text = row[row_label]
        sentences = sent_tokenize(text, 'polish')
        text = ''
        for sentence in sentences:
            tokenized_sentence = ' '.join(word_tokenize(sentence))
            text = '\n'.join([text, tokenized_sentence])
        yield text


def tokened_data_from_txt(contents_files: Iterable[TextIO]) -> Generator[str, None, None]:
    for content_file in contents_files:
        content_file = content_file.read()
        sentences = sent_tokenize(content_file, 'polish')
        text = ''
        for line in sentences:
            tokenized_sentence = ' '.join(word_tokenize(line, 'polish'))
            text = '\n'.join([text, tokenized_sentence])
        yield text


def open_files(files_path: list[str]) -> Generator[TextIO, None, None]:
    for path in files_path:
        path = os.path.abspath(path)
        with open(path) as text_file:
            yield text_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument('output_path',
                        type=str,
                        help='Path to output')

    parser.add_argument('-wiki',
                        '--wikipedia-path',
                        type=str,
                        help='Path to wikipedia dump csv')
    parser.add_argument('-fp',
                        '--files-path',
                        nargs='*',
                        help='Paths to .txt')

    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    output_path = os.path.abspath(args.output_path)
    with open(output_path, 'w+') as output_file:
        if args.wikipedia_path:
            print('Parsing wikipedia data')
            with open(args.wikipedia_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)
                wiki_text_gen = get_tokenized_text(data=csv_reader, row_label='CleanText')
                for i, article in enumerate(wiki_text_gen):
                    output_file.write(article)
                    output_file.write('\n')
                    if i % 1000 == 0:
                        print('Wikipedia: ', i)
        else:
            print('Wikipedia path not set')

        if args.files_path:
            print('Parsing .txt data')

            content_gen = open_files(args.files_path)
            tokenized_content_gen = tokened_data_from_txt(content_gen)

            for i, content in enumerate(tokenized_content_gen):
                output_file.write(content)
                output_file.write('\n')
                if i % 25 == 0:
                    print('.txt data write iteration: ', i)
        else:
            print('.txt path not set')
