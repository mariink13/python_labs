# python_labs
## Лабораторная работа 1

### Задание 1
```python
name=str(input("Имя: "))
age=int(input("Возраст: "))

print("Привет, " + name + "! Через год тебе будет " + str(age + 1) + ".")
```
![Картинка 1](./src/lab01/img/ex01_img.png)

### Задание 2
```python
a = float(input('a: ').replace(',','.'))
b = float(input('b: ').replace(',','.'))
print(f"sum={round(a+b,2)}; avg={round((a+b)/2,2)}")
```

![Картинка 2](/![Снимок экрана 2025-09-24 в 11.56.29.png](scr/lab01/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202025-09-24%20%D0%B2%2011.56.29.png))

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
![Картинка 3](/images/ex03_img.png)

### Задание 4
```python
m=int(input("Минуты: "))
h=m//60
m1=m%60
print(f"{h}:{m1:02d}")
```
![Картинка 4]()

### Задание 5
```python
n=str(input(("Введите фИО: ")))
n=" ".join(n.split())
p= n.split()
if len(p)==3:
    surname, name, patr = p
initials=f"{surname[0]}{name[0]}{patr[0]}."
l=len(n)
print("Инициалы:" , initials)
print("Длина(символов):" , l)
```
![Картинка 5](/![ex05_img.png](src/lab01/img/ex05_img.png))