import numpy
import os       # файлы
import glob     # Спиок файлов
import subprocess   # Запуск .exe

# poseModel = op.PoseModel.BODY_25
# print(op.getPoseBodyPartMapping(poseModel))
# print(op.getPoseNumberBodyParts(poseModel))
# print(op.getPosePartPairs(poseModel))
# print(op.getPoseMapIndex(poseModel))

# bin\OpenPoseDemo.exe --video [examples\media\video.avi] --face --hand --write_json [output_json_folder/]
fail = 0
folders = os.listdir('G:\\data_set\\database\\weizman')
print('Folders with video:', folders)
i = 27
count = 0
for i in folders:
    print('\t', i)
    pathjson = "G:\\data_set\\summer\\json" + '\\' + i  # Папка для JSON файлов одной группы видео
    # Создание папки для группы действий...
    if not os.path.exists(pathjson):
        os.mkdir(pathjson)
    print('!!!!!!!!!!' + pathjson)
    path = "G:\\data_set\\database\\weizman" + '\\' + i    # Папка c видео одной группы
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
        if not os.listdir(pathjson1video):  # Создание JSON файла для видео j
            res = r'bin\OpenPoseDemo.exe --video ' + path + '\\' + j + ' --write_json ' + pathjson1video
            print(res)
            try:
                subprocess.check_call(res)
            except Exception:
                print("Ошибка")

        else:
            count += 1
            print("Already exist")

print("end")
print(count)
