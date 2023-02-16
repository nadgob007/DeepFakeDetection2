import os       # файлы
import glob     # Спиок файлов
import subprocess   # Запуск .exe
import json         # Для JSON файлов
from pprint import pprint   # Подключили Pprint для красоты выдачи текста

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re


def cutcolor(listi, color):
    result = ['black']
    for i in color:
        result = re.findall(listi, i)
        if len(result) != 0:
            if result[0] == listi:
                color.remove(i)
            print(result)
    return color


def dots_wz(a, b, array):
    tmp = 0
    flag = True
    count = 0
    for i in array:
        if flag:
            tmp = i
            flag = False
        else:
            if tmp != 0 and i != 0:
                a.append(tmp)
                b.append(i)
            flag = True
        count += 1
    return count

def dots(a, b, array):
    flag = True
    count = 0
    for i in array:
        if flag:
            a.append(i)
            flag = False
        else:
            b.append(i)
            flag = True
        count += 1
    return count


# array1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# array = [1, 2, 3, 0, 5, 6, 0, 8, 9, 9, 0, 0, 6, 5, 4, 3, 2, 1]
array = [1, 1, 9, 1]

x = []
y = []
count = dots(x, y, array)
print(x)
print(y)
print(count)

color_list = []
for name, hex in matplotlib.colors.cnames.items():
    color_list.append(name)
print(color_list)

print(len(color_list))

color_list = cutcolor('gray', color_list)
color_list = cutcolor('grey', color_list)
color_list = cutcolor('white', color_list)
color_list = cutcolor('light', color_list)
color_list = cutcolor('snow', color_list)
color_list = cutcolor('azure', color_list)
color_list = cutcolor('ivory', color_list)
color_list = cutcolor('ivory', color_list)
color_list = cutcolor('alice', color_list)
color_list = cutcolor('beige', color_list)
color_list = cutcolor('bisque', color_list)
color_list = cutcolor('blanchedalmond', color_list)
color_list = cutcolor('cornsilk', color_list)

print(color_list)
print(len(color_list))

num = 0
fig, ax = plt.subplots()
for i in color_list:
    ax.plot(x, y, color=i)     # Отрисовка значений на графике
    ax.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
    ax.set_title('Difference: $hand coordinate$ versus $legs coordinate$')  # Название над графиком
    ax.set_xlabel('$legs(x)$')  # Название оси x
    ax.set_ylabel('$hands(y)$') # Название оси Y
    fig.tight_layout()
    y.insert(0, y[1]+1)
    y.pop(1)
    y.insert(1, y[1]+1)
    y.pop(2)
    print(num)
    num += 1
    plt.show()

plt.show()
