import re
from collections import Counter


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру и нормализует пробелы."""
    if not text:
        return ""

    # Приводим к нижнему регистру, сохраняя букву "ё"
    text = text.lower()

    # Заменяем все пробельные символы на обычные пробелы
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы в начале и конце
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Разбивает текст на слова (токены)."""
    if not text:
        return []

    # Используем регулярное выражение для извлечения слов (включая цифры и дефисы)
    tokens = re.findall(r"\b[\w-]+\b", text, re.IGNORECASE)

    # Приводим все токены к нижнему регистру для консистентности
    return [token.lower() for token in tokens]


def count_freq(tokens: list[str]) -> dict[str, int]:
    """Подсчитывает частоту слов."""
    if not tokens:
        return {}

    # Приводим все токены к нижнему регистру для case-insensitive подсчета
    lower_tokens = [token.lower() for token in tokens]
    return dict(Counter(lower_tokens))


def top_n(freq: dict[str, int], n: int) -> list[tuple[str, int]]:
    """Возвращает n самых частых слов с их частотами."""
    if not freq or n <= 0:
        return []

    # Сортируем по убыванию частоты, при равной частоте - по алфавиту
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    return sorted_items[:n] if n > 0 else []
