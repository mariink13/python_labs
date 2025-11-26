# Временный скрипт для проверки CSV
import csv
from io import StringIO

# Тестируем твой проблемный CSV
csv_content = '"unclosed,quote\n'
try:
    reader = csv.DictReader(StringIO(csv_content))
    data = list(reader)
    print(f"Данные: {data}")
    print(f"Количество строк: {len(data)}")
except Exception as e:
    print(f"Ошибка: {e}")
