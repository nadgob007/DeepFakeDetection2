import os           # файлы
import re           # Регуляоные выражения
import glob         # Спиок файлов
import numpy as np  # мат операции
from skimage import color   # Отображение изображений
from pprint import pprint   # Подключили Pprint для красоты выдачи текста
import matplotlib.pyplot as plt                 # Графики
from skimage.io import imread, imshow, show     # Отображение изображений
from sklearn.model_selection import train_test_split    # Разбиение данных на обучение и тестирования
from sklearn.neighbors import KNeighborsClassifier      # Классификация ближайших соседей
from sklearn.pipeline import make_pipeline              # Классификация векторов поддержки С
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score     # Классификатор дерева решений
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime                           # Время выполнения скрипта
import time
from module import *


def get_data_list(n, path_true, path_false):

    # true_dirpath = []
    # true_dirnames = []
    # true_filenames = []
    true_items = []
    for dirpath, dirnames, filenames in os.walk(path_true):
        if not (len(filenames) == 0) and len(dirnames) == 0:
            # true_dirpath.append(''.join(map(str, dirpath)))
            # true_dirnames.extend(dirnames)
            # true_filenames.extend(filenames)
            for i in filenames:
                true_items.append(dirpath+'\\'+i)

    # false_dirpath = []
    # false_dirnames = []
    # false_filenames = []
    false_items = []
    for dirpath, dirnames, filenames in os.walk(path_false):
        if not (len(filenames) == 0) and len(dirnames) == 0:
            # false_dirpath.append(''.join(map(str, dirpath)))
            # false_dirnames.extend(dirnames)
            # false_filenames.extend(filenames)
            for i in filenames:
                false_items.append(dirpath + '\\' + i)

    return true_items[0:int(n/2)], false_items[0:int(n/2)]


# Примнимает список имен используемых изображений сохраняет и
# возвращает x_train - вектор признаков и y_train - вектор классов
def list2psD1_2(list_allK1, path_folder):

    for i in range(0, len(list_allK1)):
        # if i == 13:
        #     break
        # Обучающий набор
        list_train = list_allK1[i].get(0)
        train_numbers = []
        y_train = []
        x_train = []
        for img in list_train:
            truth = img[0]
            y_train.append(truth)
            train_numbers.append(img[1])

        count = 0
        for j in train_numbers:
            path = j
            img, img_grey, fft2, psd1D = calculate_features(path, False)
            x_train.append(psd1D)
            print(f'Train №{i}')
            print(f'Номер: {j}')
            print(f'Прогресс: {count/8}%')
            count += 1

        # Создаем папку с номером набора 1К, сли её нет
        if not os.path.exists(path_folder + '\\' + str(i)):
            os.mkdir(path_folder + '\\' + str(i))

        # Путь до файла сохранения
        psd_save(path_folder +'\\'+ str(i) + '\\train_psd.txt', x_train, y_train, train_numbers)

        # Тестовый набор
        list_test = list_allK1[i].get(1)
        test_numbers = []
        y_test = []
        x_test = []
        for img in list_test:
            truth = img[0]
            y_test.append(truth)
            test_numbers.append(img[1])

        count = 0
        for j in test_numbers:
            path = j
            print(path)
            img, img_grey, fft2, psd1D = calculate_features(path, False)
            x_test.append(psd1D)
            print(f'Test №{i}')
            print(f'Номер: {j}')
            print(f'Прогресс: {count/2}%')
            count += 1

        # Путь до файла сохранения
        psd_save(path_folder +'\\'+ str(i) + '\\test_psd.txt', x_test, y_test, test_numbers)

    return x_train, y_train, x_test, y_test


'''
n - размер выборки
tf - train test соотношение
path_true path_false - пути до папок true false
path - куда сохранить psd

Возвращает:
    train.txt и test.txt файлы с путями до *.png файлов
'''


def data_to_psd(n, sample, tf, path_true, path_false, path):

    # 1. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
    true, false = get_data_list(n, path_true, path_false)

    # 2. помечаем 1 - true, 0 - false
    true = [[1, true[i]] for i in range(len(true))]
    false = [[0, false[i]] for i in range(len(false))]

    # 3. соединяем и перемешиваем.
    true_false = true + false
    all_train, all_test = train_test_split(true_false, train_size=tf, random_state=42)

    # 4. разбиваем массив на n/sample папок (20)
    all_K1 = []
    for j in range(int(n/sample)):
        K1_train = [all_train[i] for i in range(0 + (j * int(sample*tf)), int(sample*tf) + (j * int(sample*tf)))]
        K1_test = [all_test[i] for i in range(0 + (j * int(sample-sample*tf)), int(sample-sample*tf) + (j * int(sample-sample*tf)))]
        all_K1.append({0: K1_train, 1: K1_test})  # 0-train 1-test

    # 5. Получаем массив признаков для изображений
    x_train, y_train, x_test, y_test = list2psD1_2(all_K1, path)
    return 0


def classification(path, number_folders, interval):
    all_kn = []
    all_svm = []
    all_dt = []

    for i in range(number_folders):
        kn, svm, dt = classifier(path + '\\' + str(i), interval)
        all_kn.append(kn)
        all_svm.append(svm)
        all_dt.append(dt)
    accuracy_save(path + '\\acc.txt', all_kn, all_svm, all_dt)

    print(f'KN, SVM, DT: {kn / 2}%, {svm / 2}%, {dt / 2}%')

def classification10(path, number_of_folders):
    for j in range(number_of_folders):
        all_kn = []
        all_svm = []
        all_dt = []
        intervals = []
        for i in range(0, 720, 10):
            if i == 710:
                interval = [[i, i+14]]
            else:
                interval = [[i, i+10]]

            kn, svm, dt = classifier(path + '\\' + str(j), interval)
            all_kn.append(kn)
            all_svm.append(svm)
            all_dt.append(dt)
            intervals.append(interval[0])
        print(f'Выборка:{j}')
        save_in_1K(path +'\\'+ str(j) + '\\acc.txt', all_kn, all_svm, all_dt, intervals, mode=10)

def classification20(path, number_folders):

    for j in range(0, number_folders):
        all_kn = []
        all_svm = []
        all_dt = []
        intervals = []

        for i in range(0, 720, 10):
            for k in range(10 + i, 720, 10):
                interval = []
                interval.append([i, i + 10])

                if k == 710:
                    interval.append([k, k + 14])
                else:
                    interval.append([k, k + 10])

                kn, svm, dt = classifier(path + '\\' + str(j), interval)
                all_kn.append(kn)
                all_svm.append(svm)
                all_dt.append(dt)
                intervals.append(interval[0])
                intervals.append(interval[1])
            print(f'Выборка:{j},Интервал:{i}')

        save_in_1K(path + '\\' + str(j) + '\\acc20.txt', all_kn, all_svm, all_dt, intervals, mode=20)


# https://habr.com/ru/post/669170/
if __name__ == '__main__':
    # E:\NIRS\Frequency\Faces-HQ2\false\1m_faces_00_01_02_03\1m_faces_00\9G6G661H8A.jpg - ломаное
    # Начало
    start_time = datetime.now()

    count_of_samples = 20000    # колличество фотографий в 1 классе
    size_of_sample = 1000       # колличество фотографий в выборке
    number_of_folders = int(count_of_samples / size_of_sample)   # колличество папок по size_of_sample фотографий
    p = 0.80    # Процент тренировочной части выборки
    count_of_features = 724

    path = "E:\\NIRS\\Frequency\\Faces-HQ2"

    a = 1   # Подаются пути к данным, создаются txt файлы с psd. Использовать для перерасчёта.
    if a == 1:
        data_to_psd(count_of_samples, size_of_sample, p, path+'\\true', path+'\\false', path+'\\split')

    b = 1   # Классификация по имеющимся txt файлам
    if b == 1:
        interval = [[0, count_of_features]]
        classification(path + '\\split', number_of_folders, interval)
    # Конец
    print(datetime.now() - start_time)
    c = 1  # Отображение данных классификаторов
    if c == 1:
        kn_all, svm_all, dt_all = read_acc(path + '\\split\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all)

    d = 0  # Перещёт классификаторами для интервала в 20 признаков из участков по 10 из разных частей вектора признаков.
    if d == 1:
        classification20(path + '\\split', number_of_folders)

    f = 0  # Перещёт классификаторами для интервала в 10 признаков
    if f == 1:
        #classification10(path + '\\split', number_of_folders)
        kn_all, svm_all, dt_all = read_acc(path + '\\split\\0\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all)

    e = 0  # Отображение тепловой карты
    if e == 1:
        all_kn, all_svm, all_dt, intervals = read_acc20(path + '\\split', number_of_folders)
        show_temp(all_kn, all_svm, all_dt, intervals, number_of_folders)


    # Конец
    print(datetime.now() - start_time)
