n=str(input(("Введите фИО: ")))
n=" ".join(n.split())
p= n.split()
if len(p)==3:
    surname, name, patr = p
initials=f"{surname[0]}{name[0]}{patr[0]}."
l=len(n)
print("Инициалы:" , initials)
print("Длина(символов):" , l)



