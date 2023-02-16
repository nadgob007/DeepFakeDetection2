import os  # файлы
import re
import glob  # Спиок файлов
import subprocess  # Запуск .exe
import json  # Для JSON файлов
import matplotlib # Для отрисовки графика
import matplotlib.pyplot as plt # Для отрисовки графика
import numpy as np  # Для создания массивов
from pprint import pprint  # Подключили Pprint для красоты выдачи текста
from sklearn.decomposition import PCA   # Анализ главных компонент (pip install scikit-learn)
import umap.umap_ as umap # (pip install umap-learn)


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


folders = os.listdir('G:\\data_set\\summer\\tables')

points_graph = []
color_num = 3
for folder in folders:
    table = "G:\\data_set\\summer\\tables" + '\\' + folder

    actions = os.listdir(table)
    num = 0
    print(folder)

    fig2, (ax3, ax4) = plt.subplots(  # Объект для графика группы действий одного класса
        nrows=1, ncols=2,
        figsize=(8, 4)
    )  # Пространство графика

    for action in actions:
        empty = False  # Если текстовый файл пустой

        table_actions = table + '\\' + action

        f = open(table_actions + '\\table.txt', 'r')

        line = f.readline()
        dline = 0  # Если первая трока в файле пустая

        lines = []
        array = np.array([0])  # Не сжатый массив
        array = np.delete(array, 0)
        j = 0
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

            if len(points) == 0:
                break
            print(points)

            array = np.append(array, points)
            array = array.reshape(((j + 1), len(points))) # Собираем точки в масив numpy

            j += 1
            dline += 1
        if empty:
            continue
        f.close()  # Закрыли файл

        # Метод уменьшения размерности PCA
        pca = PCA(n_components = 2)
        array_pca = pca.fit_transform(array)
        print(array_pca)
        print('Колличество кадров в видео:' + str(len(array_pca)))

        # Метод уменьшения размерности UMAP
        reducer = umap.UMAP()
        array_umap = reducer.fit_transform(array)
        print(array_umap)
        print('Колличество кадров в видео:' + str(len(array_umap)))

        vide_graph = []  # Для отрисовки одного видео
        vide_graph = array_pca

        # Заполняем массивы для отрисовки
        x = []
        y = []
        vide_graph = vide_graph.transpose()
        i = 0
        while i < len(vide_graph[0]):
            x.append(float(vide_graph[0][i]))
            y.append(float(vide_graph[1][i]))
            i += 1

        a = []
        b = []
        vide_graph = array_umap
        vide_graph = vide_graph.transpose()
        i = 0
        while i < len(vide_graph[0]):
            a.append(float(vide_graph[0][i]))
            b.append(float(vide_graph[1][i]))
            i += 1

        flag = True
        print(len(vide_graph))
        print(vide_graph)
        print('Print X', x)
        print('Print Y', y)

        color_list = []
        for name, hex in matplotlib.colors.cnames.items():
            color_list.append(name)

        # убираем плохо различимые цвета
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

        # Пространство графика
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2,
            figsize=(8, 4)
        )

        ax1.plot(x, y, color= color_list[num])  # Отрисовка значений на графике
        #ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
        ax1.set_title('$PCA$')  # Название над графиком
        ax1.set_xlabel('$X1$')  # Название оси x
        ax1.set_ylabel('$Y1$')  # Название оси Y
        if num < 10:
            ax3.plot(x, y, color=color_list[num + 2])  # Отрисовка значений на графике
            #ax3.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')

        #count = dots_wz(a, b, vide_graph)  # Без нулевых точек

        ax2.plot(a, b, color=color_list[num])
        #ax2.scatter(x=a, y=b, marker='o', c='b', edgecolor='r')
        ax2.set_title('$UMAP$')
        ax2.set_xlabel('$X1$')  # Название оси x
        ax2.set_ylabel('$Y1$')  # Название оси Y
        ax2.yaxis.tick_right()
        if num < 10:
            ax4.plot(a, b, color=color_list[num + 2])
            #ax4.scatter(x=a, y=b, marker='o', c='b', edgecolor='r')

        fig.tight_layout()
        plt.close('all')
        print(num)
        num += 1
        graphic = str(num)
        fig.savefig(table_actions + '\\' + action)

    ax3.set_title('$PCA$')  # Название над графиком
    ax3.set_xlabel('$X1$')  # Название оси x
    ax3.set_ylabel('$Y1$')  # Название оси Y
    ax4.set_title('$UMAP$')
    ax4.yaxis.tick_right()
    fig2.tight_layout()
    fig2.savefig('G:\\data_set\\summer\\Graphic' + '\\' + str(color_num - 3))

    print(color_num)
    color_num += 1



