import os       # файлы
import glob     # Спиок файлов
import subprocess   # Запуск .exe
import json         # Для JSON файлов
from pprint import pprint   # Подключили Pprint для красоты выдачи текста

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re

# Убираем цвета из списка
def cutcolor(listi, color):
    result = ['black']
    for i in color:
        result = re.findall(listi, i)
        if len(result) != 0:
            if result[0] == listi:
                color.remove(i)
            print(result)
    return color


# Функция заполнения массивов a и b и array без пропуска нулевых точек
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


# Функция заполнения массивов a и b и array с пропуском нулевых точек
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

folders = os.listdir('G:\\data_set\\table')

points_graph = []
color_num = 3
for folder in folders:
    table = "G:\\data_set\\table" + '\\' + folder
    actions = os.listdir(table)

    num = 0
    print(folder)

    fig2, (ax3, ax4) = plt.subplots(     # Объект для графика группы действий одного класса
        nrows=1, ncols=2,
        figsize=(8, 4)
    )  # Пространство графика

    for action in actions:
        empty = False  # Если текстовый файл пустой

        table_actions = table + '\\' + action

        f = open(table_actions + '\\table.txt', 'r')

        line = f.readline()
        dline = 0   # Если первая трока в файле пустая

        lines = []
        while line:
            print(line)
            line = f.readline()
            if line == '' and dline == 0:
                empty = True
                break

            result = re.split(r'\t', line)
            result.pop(0)
            points = []
            for i in result:
                points.append(float(i))
            lines.extend(points)
            dline += 1
        if empty:
            continue
        f.close()   # Закрыли файл

        h1 = 9  # 4ая точка (4*2)+1
        h2 = 15  # 7ая точка (7*2)+1
        l1 = 23  # 11ая точка (11*2)+1
        l2 = 29  # 14ая точка (14*2)+1
        vide_graph = []    # Для отрисовки одного видео
        while h1 + 41 != len(lines):  # Суммарно будет равно концу
            hres = lines[h1] - lines[h2]
            lres = lines[l1] - lines[l2]

            points_graph.append(hres)
            vide_graph.append(hres)
            points_graph.append(lres)
            vide_graph.append(lres)
            h1 += 50
            h2 += 50
            l1 += 50
            l2 += 50

            color_list = []
        for name, hex in matplotlib.colors.cnames.items():
            color_list.append(name)
        
        # убмраем плохо различимые цвета
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

        x = []
        y = []
        a = []
        b = []
        flag = True
        print(len(vide_graph))
        print(vide_graph)

        count = dots(x, y, vide_graph)
        print('count', count)
        print('Print', x)
        print('Print', y)

        # Пространство графика
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2,
            figsize=(8, 4)
        )

        ax1.plot(x, y, color=color_list[num])  # Отрисовка значений на графике
        ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
        ax1.set_title('Difference: $hand coordinate$ versus $legs coordinate$')  # Название над графиком
        ax1.set_xlabel('$legs(x)$')  # Название оси x
        ax1.set_ylabel('$hands(y)$')  # Название оси Y
        if num < 10:
            ax3.plot(x, y, color=color_list[num + 2])  # Отрисовка значений на графике
            ax3.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')

        count = dots_wz(a, b, vide_graph)   # Без нулевых точек

        ax2.plot(a, b, color=color_list[num])
        ax2.scatter(x=a, y=b, marker='o', c='b', edgecolor='r')
        ax2.set_title('$without$ $zero$ $coordinates$')
        ax2.yaxis.tick_right()
        if num < 10:
            ax4.plot(a, b, color=color_list[num + 2])
            ax4.scatter(x=a, y=b, marker='o', c='b', edgecolor='r')

        fig.tight_layout()
        plt.close('all')
        print(num)
        num += 1
        graphic = str(num)

        if num != 2:
            fig.savefig(table_actions + '/' + action)

    ax3.set_title('Difference: $hand coordinate$ versus $legs coordinate$')  # Название над графиком
    ax3.set_xlabel('$legs(x)$')  # Название оси x
    ax3.set_ylabel('$hands(y)$')  # Название оси Y
    ax4.set_title('$without$ $zero$ $coordinates$')
    ax4.yaxis.tick_right()
    fig2.tight_layout()
    fig2.savefig('G:\\data_set\\Graphic' + '/' + str(color_num - 3))

    print(color_num)
    color_num += 1

