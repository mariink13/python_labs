# python_labs

## Лабораторная работа 5

### Задание A
```python
import json
import csv
from pathlib import Path


def json_to_csv(json_path: str, csv_path: str) -> None:

    json_file = Path(json_path)
    csv_file = Path(csv_path)
    
    if not json_file.exists():
        raise FileNotFoundError(f"Файл {json_path} не найден")
    
    if json_file.suffix.lower() != '.json':
        raise ValueError("Неверный тип файла. Ожидается .json")
    
    try:
        with json_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON: {e}")
    
    if not data:
        raise ValueError("Пустой JSON или неподдерживаемая структура")
    
    if not isinstance(data, list):
        raise ValueError("JSON должен содержать список объектов")
    
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Все элементы JSON должны быть словарями")
    
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    if data:
        first_item_keys = list(data[0].keys())
        remaining_keys = sorted(all_keys - set(first_item_keys))
        fieldnames = first_item_keys + remaining_keys
    else:
        fieldnames = sorted(all_keys)
    # Запись в CSV
    try:
        with csv_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                complete_row = {key: row.get(key, '') for key in fieldnames}
                writer.writerow(complete_row)
    except Exception as e:
        raise ValueError(f"Ошибка записи CSV: {e}")

def csv_to_json(csv_path: str, json_path: str) -> None:
  
    csv_file = Path(csv_path)
    json_file = Path(json_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Файл {csv_path} не найден")

    if csv_file.suffix.lower() != '.csv':
        raise ValueError("Неверный тип файла. Ожидается .csv")
    
    try:
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV файл не содержит заголовка")
            
            data = list(reader)
            
    except Exception as e:
        raise ValueError(f"Ошибка чтения CSV: {e}")

    if not data:
        raise ValueError("Пустой CSV файл")

    try:
        with json_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise ValueError(f"Ошибка записи JSON: {e}")

json_to_csv("/Users/marinaujmanova/Desktop/python_labs/src/data/samples/people.json", "/Users/marinaujmanova/Desktop/python_labs/src/data/out/people_from_json.csv")
csv_to_json("/Users/marinaujmanova/Desktop/python_labs/src/data/samples/people.csv", "/Users/marinaujmanova/Desktop/python_labs/src/data/out/people_from_csv.json")
```

### Входные данные
![exAf](./img/lab05/exAf.png)
![exAfi](./img/lab05/exAfi.png)

### Выходные данные
![exAex](./img/lab05/exAex.png)
![exAex2](./img/lab05/exAex2.png)



### Задание B
```python
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def csv_to_xlsx(csv_path: str, xlsx_path: str) -> None:

    csv_file = Path(csv_path)
    xlsx_file = Path(xlsx_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Файл {csv_path} не найден")
    
    if csv_file.suffix.lower() != '.csv':
        raise ValueError("Неверный тип файла. Ожидается .csv")
    
    # Чтение CSV
    try:
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        raise ValueError(f"Ошибка чтения CSV: {e}")
    
    if not rows:
        raise ValueError("Пустой CSV файл")
    
    if not rows[0]:
        raise ValueError("CSV файл не содержит заголовка")
    
    # Создание XLSX
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        
        for row in rows:
            ws.append(row)
        
        
        for col_idx, column_cells in enumerate(ws.columns, 1):
            max_length = 8  
            column_letter = get_column_letter(col_idx)
            
            for cell in column_cells:
                try:
                    if cell.value:

                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            
            adjusted_width = max_length + 2
            ws.column_dimensions[column_letter].width = adjusted_width
        
        wb.save(xlsx_file)
        
    except Exception as e:
        raise ValueError(f"Ошибка создания XLSX: {e}")
    
csv_to_xlsx("/Users/marinaujmanova/Desktop/python_labs/src/data/samples/people.csv", "/Users/marinaujmanova/Desktop/python_labs/src/data/out/people.xlsx")
```

### Входные данные
![exAf](./img/lab05/exAf.png)


### Выходные данные
![exB2](./img/lab04/exB2_img.png)


## Лабораторная работа 4

### Задание A
```python
import csv
from pathlib import Path
from typing import Iterable, Sequence
def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    try:
        return Path(path).read_text(encoding=encoding)
    except FileNotFoundError:
        return "Такого файла нету"
    except UnicodeDecodeError:
        return "Неудалось изменить кодировку"

def write_csv(rows: list[tuple | list], path: str | Path, header: tuple[str, ...] | None = None) -> None:
    p = Path(path)
    with p.open('w', encoding="utf-8") as file:
        f = csv.writer(file)   
        if header is None and rows == []:
            file_c.writerow(('a', 'b'))
        if header is not None:
            f.writerow(header)
        if rows != []:
            const = len(rows[0])
            for i in rows:
                if len(i) != const:
                    return ValueError
        f.writerows(rows)

def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

print(read_text(r"/Users/marinaujmanova/Desktop/python_labs/src/lab04/data/input.txt"))
write_csv([("word","count"),("test",3)], r"/Users/marinaujmanova/Desktop/python_labs/src/lab04/data/check.csv")
```
![exA1](./img/lab04/exA1_img.png)
![exA2](./img/lab04/exA2_img.png)




### Задание B
```python
from io_txt_csv import read_text, write_csv, ensure_parent_dir
import sys
from pathlib import Path

sys.path.append(r'Users/marinaujmanova/Desktop/python_labs/src/lab04/lib')

from lib.text import *

def exist_path(path_f: str):
    return Path(path_f).exists()


def main(file: str, encoding: str = 'utf-8'):
    if not exist_path(file):
        raise FileNotFoundError
    
    file_path = Path(file)
    text = read_text(file, encoding=encoding)
    norm = normalize(text)
    tokens = tokenize(norm)
    freq_dict = count_freq(tokens)
    top = top_n(freq_dict)
    top_sort = sorted(top, key=lambda x: (x[1], x[0]), reverse=True)
    report_path = file_path.parent / 'report.csv'
    write_csv(top_sort, report_path, header=('word', 'count'))
    
    print(f'Всего слов: {len(tokens)}')
    print(f'Уникальных слов: {len(freq_dict)}')
    print('Топ:')
    for cursor in top_sort:
        print(f'{cursor[0]}: {cursor[-1]}')


main(r'/Users/marinaujmanova/Desktop/python_labs/src/lab04/data/input.txt')
```
![exb1](./img/lab04/exb1_img.png)
![exB2](./img/lab04/exB2_img.png)

## Лабораторная работа 3

### Задание 1
```python
def normalize(text: str, *, casefold: bool = True, yo2e: bool = True) -> str:
    if text is None:
        raise ValueError
    if not isinstance(text, str):
        raise TypeError
    if len(text) == 0:
        return "" 
    if casefold:
        text= text.casefold()
    if yo2e:
        text=text.replace('ё', 'е').replace('Ё','Е')
    text=text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    while '  ' in text:
        text=text.replace('  ', ' ')
    text= text.strip()
    return text
print(normalize("ПрИвЕт\nМИр\t"))
print(normalize("ёжик, Ёлка")) 
print(normalize("Hello\r\nWorld"))
print(normalize("  двойные   пробелы  "))
```
![ex01](./img/lab03/ex01_img.png)

### Задание 2
```python
import re
def tokenize(text: str) -> list[str]:
    reg = r'\w+(?:-\w+)*'
    text = re.findall(reg, text)
    return text
print(tokenize("привет мир"))
print(tokenize("hello,world!!!"))
print(tokenize("по-настоящему круто"))
print(tokenize("2025 год"))
print(tokenize("emoji 😀 не слово"))
```

![ex02](./img/lab03/ex02_img.png)

### Задание 3
```python
def count_freq(tokens: list[str]) -> dict[str, int]:
    freq_dict = {}
    if not tokens:
        return {}
    for token in tokens:
        freq_dict[token] = freq_dict.get(token, 0) +1
    return freq_dict
print(count_freq(["a","b","a","c","b","a"]))
print(count_freq(["bb","aa","bb","aa","cc"]))
```
![ex03](./img/lab03/ex03_img.png)

### Задание 4
```python
def top_n(freq: dict[str, int], n: int = 5) -> list[tuple[str, int]]:
    if not freq:
        return []
    items = list(freq.items())
    items.sort(key=lambda x: x[0])           # Сортировка по слову A→Z
    items.sort(key=lambda x: x[1], reverse=True)  # Сортировка по частоте 9→0
    return items[:n]
freq1 = {"a": 3, "b": 2, "c": 1}
print(top_n(freq1, 2))
freq2 = {"bb": 2, "aa": 2, "cc": 1}
print(top_n(freq2, 2))
```
![ex04](./img/lab03/ex04_img.png)

### Задание B
```python

import sys
from lib.e11_tex_stats import normalize, tokenize, count_freq, top_n
def main():
    text = sys.stdin.buffer.read().decode('utf-8') #вход к бинарным данным,преобразует строку в юникод
    if not text.strip():
        raise ValueError('Нет текста :(')
    normalized_text = normalize(text)
    tokens = tokenize(normalized_text)
    

    if not tokens:
        print("В тексте не найдено слов")
        raise ValueError

    total_words = len(tokens) # общее количество слов
    freq_dict = count_freq(tokens) # словарь частот
    unique_words = len(freq_dict) # количеситво уникальных слов 
    top_words = top_n(freq_dict, 5) # самые популярные частоты
    
    print(f"Всего слов: {total_words}")
    print(f"Уникальных слов: {unique_words}")
    print("Топ-5:")
    for word, count in top_words:
        print(f"{word}: {count}")


if name == "__main__":  
    main()
```

## Лабораторная работа 2

### Задание 1
```python
def min_max(nums: list[float | int]) -> tuple[float | int, float | int]:
    if not nums:
        raise ValueError("Пустой список!!!")
    m1=min(nums)
    m2=max(nums)
    return(m1, m2)
print(min_max([3, -1, 5, 5, 0]))
print(min_max([42]))
print(min_max([-5, -2, -9]))
print(min_max([1.5, 2, 2.0, -3.1]))
print(min_max([]))
```
![ex01](./img/lab02/ex01_img.png)

### Задание 2
```python
def unique_sorted(nums: list[float | int]) -> list[float | int]:
    return sorted(set(nums))
    
print(unique_sorted([3, 1, 2, 1, 3]))
print(unique_sorted([]))
print(unique_sorted([-1, -1, 0, 2, 2]))
print(unique_sorted([1.0, 1, 2.5, 2.5, 0]))
```

![ex02](./img/lab02/ex02_img.png)


### Задание 3
```python
def flatten(mat: list[list | tuple]) -> list:
    result=[]
    for row in mat:
        if isinstance(row, (list, tuple)): #проверяю является ли row списком или кортежем
            result.extend(row) #добавляю элементы row по отдельгости в новый список
        else:
            raise TypeError
    return (result)
    
print(flatten([[1, 2], [3, 4]])) 
print(flatten([[1, 2], (3, 4, 5)]))
print(flatten([[1], [], [2, 3]]))
print(flatten([[1, 2], "ab"]))
```
![ex03](./img/lab02/ex03_img.png)

### Задание 4
```python
def transpose(mat: list[list[float | int]]):
    if not mat:
        return []
    rows=len(mat)
    cols=len(mat[0])
    for row in mat:
        if len(row)!=cols:
            raise ValueError 
        
    new_mat = [[mat[i][j] for i in range(rows)] for j in range(cols)]

    return new_mat
print(transpose([[1, 2, 3]]))
print(transpose([[1], [2], [3]]))
print(transpose([[1, 2], [3, 4]]))
print(transpose([]))
print(transpose([[1, 2], [3]]))
```
![ex04](./img/lab02/ex04_img.png)

### Задание 5
```python
def row_sums(mat: list[list[float | int]]):
    if not mat:
        return []
    rows= len(mat)
    cols= len(mat[0])
    for row in mat:
        if len(row)!=cols:
            raise ValueError
    sums=[sum(row) for row in mat]
    return sums
print(row_sums([[1,2,3], [4,5,6]]))
print(row_sums([[-1, 1], [10, -10]]))
print(row_sums([[0,0], [0,0]]))
print(row_sums([[1,2], [3]]))
```
![ex05](./img/lab02/ex05_img.png)

### Задание 6
```python
def col_sums(mat: list[list[float | int]]):
    if not mat:
        return []
    rows = len(mat)
    cols = len(mat[0])
    for row in mat:
        if len(row) != cols:
            raise ValueError
    sums = [sum(mat[i][j] for i in range(rows)) for j in range(cols)]
    return sums
print(col_sums([[1, 2, 3], [4, 5, 6]]))
print(col_sums([[-1, 1], [10, -10]]))
print(col_sums([[0, 0], [0, 0]]))
print(col_sums([[1, 2], [3]]))
```
![ex06](./img/lab02/ex06_img.png)

### Задание 7
```python
def format_record(student: tuple[str, str, float]) -> str:
    if len(student) != 3: 
        raise ValueError("Не хватает данных")
    if not (isinstance(student[0], str) and isinstance(student[1], str) and isinstance(student[2], float)): 
        raise TypeError
    fio_parts = student[0].split() 
    if len(fio_parts) < 2:
        raise ValueError("ФИО должно содержать фамилию и имя")
    fio_parts = [part.strip() for part in fio_parts if part.strip()]
    res = fio_parts[0].title() + " " + fio_parts[1][0].upper()
    if len(fio_parts) == 3:
        res += "." + fio_parts[2][0].upper() 
        res += "., "  
    res += " гр. " + student[1] + ", GPA " + f"{round(student[2],2):.2f}" 
    return res 
print(format_record(("Иванов Иван Иванович","BIVT-25",4.6)))
print(format_record(("Петров Пётр", "IKBO-12", 5.0)))
print(format_record(("Петров Пётр Петрович", "IKBO-12", 5.0)))
print(format_record((" сидорова  анна   сергеевна ", "ABB-01", 3.999)))
print(format_record(("Иванов Иван Иванович","BIVT-25")))
```
![ex07](./img/lab02/ex07_img.png)




## Лабораторная работа 1

### Задание 1
```python
name=str(input("Имя: "))
age=int(input("Возраст: "))

print("Привет, " + name + "! Через год тебе будет " + str(age + 1) + ".")
```
![ex01](./img/lab01/ex01_img.png)

### Задание 2
```python
a = float(input('a: ').replace(',','.'))
b = float(input('b: ').replace(',','.'))
print(f"sum={round(a+b,2)}; avg={round((a+b)/2,2)}")
```

![ex02](./img/lab01/ex02_img.png)

### Задание 3
```python
price=int(input())
discount=int(input())
vat=float(input())
base= price * (1 - discount/100)
vat_amount = base * (vat/100)
total= base + vat_amount
print("База после скидки:", base)
print("НДС:", vat_amount)
print("Итого к оплате:",total)
```
![ex03](./img/lab01/ex03_img.png)

### Задание 4
```python
m=int(input("Минуты: "))
h=m//60
m1=m%60
print(f"{h}:{m1:02d}")
```
![ex04](./img/lab01/ex04_img.png)

### Задание 5
```python
sec, fir, thr = map(str, input("ФИО: ").split())
print(f'Инициалы: {sec[0] + fir[0] + thr[0]}')
print(f'Длина(символы): {2+len(fir) + len(thr) + len(sec)}')

```
![ex05](./img/lab01/ex05_img.png)


