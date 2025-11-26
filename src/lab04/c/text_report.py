from io_txt_csv import read_text, write_csv, ensure_parent_dir
import sys
from pathlib import Path

sys.path.append(r"Users/marinaujmanova/Desktop/python_labs/src/lab04/lib")

from lib.text import *


def exist_path(path_f: str):
    return Path(path_f).exists()


def main(file: str, encoding: str = "utf-8"):
    if not exist_path(file):
        raise FileNotFoundError

    file_path = Path(file)
    text = read_text(file, encoding=encoding)
    norm = normalize(text)
    tokens = tokenize(norm)
    freq_dict = count_freq(tokens)
    top = top_n(freq_dict)
    top_sort = sorted(top, key=lambda x: (x[1], x[0]), reverse=True)
    report_path = file_path.parent / "report.csv"
    write_csv(top_sort, report_path, header=("word", "count"))

    print(f"Всего слов: {len(tokens)}")
    print(f"Уникальных слов: {len(freq_dict)}")
    print("Топ:")
    for cursor in top_sort:
        print(f"{cursor[0]}: {cursor[-1]}")


main(r"/Users/marinaujmanova/Desktop/python_labs/src/lab04/data/input.txt")
