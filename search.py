import numpy
import os       # файлы
import glob     # Спиок файлов
import subprocess   # Запуск .exe


def copy_and_create_json(destpath, initpath):
    fail = 0
    folders = os.listdir(initpath)    # i
    print('Folders with video:', folders)
    i = 27
    count = 0
    for i in folders:
        print('\t', i)
        pathjson = destpath + '\\' + i  # d Папка для JSON файлов одной группы видео
        # Создание папки для группы действий...
        if not os.path.exists(pathjson):
            os.mkdir(pathjson)
        print('!!!!!!!!!!' + pathjson)
        path = initpath + '\\' + i    # i Папка c видео одной группы
        print(path)
        filesAVI = [x for x in os.listdir(path) if x.endswith(".avi")]  # Список файлов .avi ,которые лежат в одной папке
        print(filesAVI)
        # Создание JSON файлов одного видео
        for j in filesAVI:
            str_list = list(j)
            for num in [-1, -2, -3, -4]:
                str_list[num] = ''
            jmk = "".join(str_list)
            print(j)
            pathjson1video = pathjson + '\\' + jmk  # Путь к папке
            # Создание папки JSON файлов одного видео
            if not os.path.exists(pathjson1video):
                os.mkdir(pathjson1video)

                # Создание JSON файла для видео j
                res = r'bin\OpenPoseDemo.exe --video ' + path + '\\' + j + ' --write_json ' + pathjson1video
                print(res)
                try:
                    subprocess.check_call(res)
                except Exception:
                    print("Ошибка")

            else:
                count += 1
                print("Already exist")
    return 0


initpath = "G:\\data_set\\hmdb51_sta"
destpath = "G:\\data_set\\p1"
copy_and_create_json(destpath, initpath)

print("end")
print(count)