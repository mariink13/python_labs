sec, fir, thr = map(str, input("ФИО: ").split())
print(f'Инициалы: {sec[0] + fir[0] + thr[0]}')
print(f'Длина(символы): {2+len(fir) + len(thr) + len(sec)}')




