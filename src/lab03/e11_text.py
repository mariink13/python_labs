import sys
from lib.e11_text_stats import normalize, tokenize, count_freq, top_n


def main():
    text = sys.stdin.buffer.read().decode(
        "utf-8"
    )  # вход к бинарным данным,преобразует строку в юникод
    if not text.strip():
        raise ValueError("Нет текста :(")
    normalized_text = normalize(text)
    tokens = tokenize(normalized_text)

    if not tokens:
        print("В тексте не найдено слов")
        raise ValueError

    total_words = len(tokens)  # общее количество слов
    freq_dict = count_freq(tokens)  # словарь частот
    unique_words = len(freq_dict)  # количеситво уникальных слов
    top_words = top_n(freq_dict, 5)  # самые популярные частоты


if name == "__main__":
    main()
