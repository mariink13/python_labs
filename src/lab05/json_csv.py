import json
import csv
from pathlib import Path


def json_to_csv(src_path: str, dst_path: str) -> None:
    """Конвертирует JSON файл в CSV."""
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(f"Файл {src_path} не найден")

    # Читаем JSON
    with open(src, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Некорректный JSON: {e}")

    if not data:
        raise ValueError("JSON файл пуст")

    # Получаем все возможные ключи
    all_keys = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("JSON должен содержать массив объектов")
        all_keys.update(item.keys())

    # Записываем CSV
    with open(dst, "w", encoding="utf-8", newline="") as f:
        if all_keys:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(data)


def csv_to_json(src_path: str, dst_path: str) -> None:
    """Конвертирует CSV файл в JSON."""
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(f"Файл {src_path} не найден")

    # Читаем CSV
    data = []
    with open(src, "r", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f)
            for row in reader:
                # Конвертируем числовые значения
                processed_row = {}
                for key, value in row.items():
                    # Проверяем что value не None и является цифрой
                    if value is not None and value.isdigit():
                        processed_row[key] = int(value)
                    else:
                        processed_row[key] = value
                data.append(processed_row)
        except csv.Error as e:
            raise ValueError(f"Некорректный CSV: {e}")

    if not data:
        raise ValueError("CSV файл пуст")

    # Записываем JSON
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
