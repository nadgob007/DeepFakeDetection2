# from __future__ import print_function
# import cv2 as cv
# import argparse
import numpy
import os  # файлы
import glob  # Спиок файлов
import subprocess  # Запуск .exe
import json  # Для JSON файлов
from pprint import pprint  # Подключили Pprint для красоты выдачи текста


folders = os.listdir('G:\\data_set\\json')
print('Folders with video:', folders)
for folder in folders:

    all_actions = "G:\\data_set\\json" + '\\' + folder  # Папка [действий] с папками [видео]
    # Создание папки для группы действий...
    table = "G:\\data_set\\table" + '\\' + folder
    if not os.path.exists(table):
        os.mkdir(table)

    actions = os.listdir(all_actions)
    for action in actions:
        specific_actions = all_actions + '\\' + action
        table_actions = table + '\\' + action

        if not os.listdir(specific_actions):    # Пропустить папку с видео, если в папке нет файлов
            print('ERROR->Video:[' + specific_actions + '] not exist')
            continue
        if not os.path.exists(table_actions):   # Не создаём папку, если она уже существует
            os.mkdir(table_actions)

        # Формируем шапку таблицы в table.txt
        f = open(table_actions + '\\table.txt', 'w')
        form = 'Frame[№]\t'
        for a in range(0, 25):
            form = form + str(a) + '\t\t'
        form += '\n'
        f.write(form)
        f.close()

        files_json = [x for x in os.listdir(specific_actions) if x.endswith(".json")]   # Список файлов посчитанных кадров

        frame_num = 0
        for file in files_json:
            path = specific_actions + '\\' + file
            with open(path, 'r', encoding='utf-8') as f:  # открыли файл с данными
                text = json.load(f)  # загнали все, что получилось в переменную
                pprint(text)  # вывели результат на экран (Опционально)
                if len(text['people']) == 1:
                    for i in text['people']:
                        j = 0
                        p = 0
                        line = 'Frame[' + str(frame_num) + ']'
                        f = open(table_actions + '\\table.txt', 'a')
                        while j != len(i['pose_keypoints_2d']):
                            x = i['pose_keypoints_2d'][j]
                            y = i['pose_keypoints_2d'][j+1]
                            c = i['pose_keypoints_2d'][j+2]
                            line = line + '\t' + str(x) + '\t' + str(y)
                            # print('Point[', p, ']', '(', x, ',', y, ')')    # Проверка вывода (Опционально)
                            j += 3
                            p += 1
                        line += '\n'
                        f.write(line)
                        print(line)     # Проверка (Опционально)
                f.close()
            frame_num += 1


