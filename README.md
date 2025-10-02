# python_labs
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