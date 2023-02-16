import os  # файлы
import glob  # Спиок файлов
import subprocess  # Запуск .exe
import json  # Для JSON файлов
from pprint import pprint  # Подключили Pprint для красоты выдачи текста
from sklearn.decomposition import PCA   # Анализ главных компонент

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re

line = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
result = np.array([0])
result = np.delete(result, 0)
j = 0
while j < 5:
    result = np.append(result, line)
    result = result.reshape(((j+1),len(line)))
    j+=1
print(result)

# Сжатие массива с помощью PCA
array = np.array(result) # Не сжатый массив
pca_model = PCA(n_components = 2)
array3 = pca_model.fit(array.transpose())
print(array3)
array2= pca_model.fit_transform(array.transpose())
print(array2)

x = []
y = []

array2 = array2.transpose()
i = 0
while i < len(array2[0]):
    x.append(float(array2[0][i]))
    y.append(float(array2[1][i]))
    i+=1

print(x)
print(y)


# Отрисовка масссива
fig, ax = plt.subplots()
plt.show()
ax.plot(x, y, color = 'aquamarine')     # Отрисовка значений на графике
ax.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax.set_title('Difference: $hand coordinate$ versus $legs coordinate$')  # Название над графиком
ax.set_xlabel('$legs(x)$')  # Название оси x
ax.set_ylabel('$hands(y)$') # Название оси Y
fig.tight_layout()
# y.insert(0, y[1]+1)
# y.pop(1)
# y.insert(1, y[1]+1)
# y.pop(2)

plt.show()
