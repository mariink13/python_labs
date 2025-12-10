from group import Group
from src.lab08.models import Student

def print_students(title, students):
    print("\n" + title)
    for s in students:
        print(f"{s.fio} | {s.birthdate} | {s.group} | {s.gpa}")

g = Group("students.csv")

print_students("Изначальный CSV:", g.list())

new_st = Student("Уйманова Марина Павловна", "2007-08-31", "БИВТ-25-6", 4.6)
g.add(new_st)
print_students("После добавления:", g.list())

found = g.find("те")
print_students("Поиск 'те':", found)

g.update("Иванов Иван Иванович", gpa=4.1, group="БИВТ-25-6")
print_students("После обновления данных Иванова:", g.list())

g.remove("Гадалова Валентина Никитовна")
print_students("После удаления Гадаловой:", g.list())
