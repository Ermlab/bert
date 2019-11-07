import argparse
import os
import sys
import tempfile
from os import cpu_count
from time import sleep
from typing import TextIO

import psutil
from multiprocessing_on_dill.pool import Pool
from progress.counter import Counter

CMD = """python {create_pretraining_data_script_path} \
  --input_file={input_file} \
  --output_file={output_file} \
  --vocab_file={vocab_file} \
  --do_lower_case={lower_case} \
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
    lines = ""
    for line in txt_file:
        if line == "\n" and lines != "":
            yield lines
            lines = ""
        elif line != "\n":
            lines += line
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
    lines = ""
    for i, line in enumerate(txt_file):
        if not i % split_after:
            yield lines
            lines = ""
        else:
            lines += line
    yield lines


def create_tf_record(
        article_text: str, output_dir: str, vocabulary_file_path: str, temporary_dir: str,
        pretraining_data_script_path: str, config_args: argparse.Namespace, no: int
) -> None:
    """Spawns a subshell and runs script creating tfrecord in it.
    Function to be called in subprocess.

    Parameters
    ----------
    article_text
        Text of the article that will be written in a temporary file.
    output_dir
        Path to the directory where tf-records will be written.
    vocabulary_file_path
        Path to the parsed vocabulary file.
    temporary_dir
        Path to the directory where article temporary file will be created.
    pretraining_data_script_path
        Path to the BERT create_pretraining_data script.
    config_args
        Args from argparse.
    no
        Sequence number of the file.
    """

    with tempfile.NamedTemporaryFile(mode="w+", dir=temporary_dir) as tmp_file:
        tmp_file.write(article_text)
        tmp_file.seek(0)
        os.system(
            CMD.format(
                input_file=tmp_file.name,
                output_file=os.path.join(output_dir, f"bert_dataset.tfrecords{no}"),
                vocab_file=vocabulary_file_path,
                bptt=config_args.bptt,
                create_pretraining_data_script_path=pretraining_data_script_path,
                lower_case=config_args.lower_case,
            )
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument("corpus_path", type=str, help="Path to corpus .txt file")
    parser.add_argument("vocab_path", type=str, help="Path to vocab file")
    parser.add_argument(
        "output_path", type=str, help="Path to folder where tfrecords will be saved"
    )
    parser.add_argument(
        "create_pretraining_data_script_path",
        type=str,
        help="Path to script creating pretraining data",
    )
    parser.add_argument("-lc", "--lower_case", action="store_true", help="Do lower case tfrecords")
    parser.add_argument(
        "-sl",
        "--split_by_lines",
        type=int,
        help="Changes default corpus splitting by blank line to by line number specify by this parameter."
             " Leave blank to use splitting by blank line",
    )
    parser.add_argument(
        "-tmp_dir_path",
        "--temporary_directory_path",
        type=str,
        default="./",
        help="Path to folder where temporary files will be saved",
    )
    parser.add_argument("--bptt", type=int, default=128, help="Max sequence length")
    parser.add_argument(
        "-cp", "--checkpoint", type=int, default=0, help="Run from specific iteration number"
    )
    args = parser.parse_args()

    EXTRACTION_FACTORY = {
        "blank": extract_article_by_blank_lines,
        "lines": extract_article_by_number_of_sentence,
    }

    corpus_path = os.path.abspath(args.corpus_path)
    create_pretraining_data_script_path = os.path.abspath(args.create_pretraining_data_script_path)
    vocab_path = os.path.abspath(args.vocab_path)
    output_path = os.path.abspath(args.output_path)
    temporary_directory_path = os.path.abspath(args.temporary_directory_path)
    checkpoint = args.checkpoint

    with open(corpus_path, "r") as corpus_file, tempfile.TemporaryDirectory(
            dir=temporary_directory_path
    ) as tmp_file_dir, Pool(processes=cpu_count()) as pool:
        if args.split_by_lines:
            print("Splitting by number of lines")
            articles = extract_article_by_number_of_sentence(corpus_file, args.split_by_lines)
        else:
            print("Splitting by blank lines")
            articles = extract_article_by_blank_lines(corpus_file)
        results = []

        for i, article in Counter("Spawning threads...").iter(enumerate(articles)):
            if i < checkpoint:
                continue

            while psutil.virtual_memory().free * 0.9 < psutil.Process(
                    os.getpid()
            ).memory_full_info().rss / (i + 1):
                print(
                    f"""Waiting 5s for RAM to be freed. 
                Currently {psutil.virtual_memory().percent}% of RAM is used."""
                )
                sleep(5)
                print([res.get() for res in results])

            f_args = (
                article,
                output_path,
                vocab_path,
                tmp_file_dir,
                create_pretraining_data_script_path,
                args,
                i,
            )
            results.append(
                pool.apply_async(func=create_tf_record, args=f_args, )
            )
        for result in results:
            while psutil.virtual_memory().free * 0.9 < psutil.Process(
                    os.getpid()
            ).memory_full_info().rss / len(results):
                print(
                    f"""Waiting 5s for RAM to be freed. 
                Currently {psutil.virtual_memory().percent}% of RAM is used."""
                )
                sleep(5)
            print(result.get())
