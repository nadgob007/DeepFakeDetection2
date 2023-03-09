from functions import *

"""
     сценарий для полной обработки (БПФ, Азимутальное усреднение)
"""
def scenario1 (initial_params):
    path = initial_params['path']

    a = 1  # Подаются пути к данным, создаются txt файлы с psd. Использовать для перерасчёта.
    if a == 1:
        data_to_psd(initial_params['all_images'], initial_params['size_of_sample'], initial_params['p'], path + '\\true', path + '\\false', path + '\\split')

    b = 1  # Классификация по имеющимся txt файлам
    if b == 1:
        interval = [[0, initial_params['count_of_features']]]
        classification(path + '\\split', initial_params['number_of_folders'], interval)

    c = 0  # Отображение данных классификаторов
    if c == 1:
        kn_all, svm_all, dt_all = read_acc(initial_params['path'] + '\\split\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all)

    d = 1  # Перещёт классификаторами для интервала в 20 признаков из участков по 10 из разных частей вектора признаков.
    if d == 1:
        classification20(path + '\\split', initial_params['number_of_folders'])

    e = 0  # Перещёт классификаторами для интервала в 10 признаков
    if e == 1:
        # classification10(path + '\\split', number_of_folders)
        kn_all, svm_all, dt_all = read_acc(path + '\\split\\0\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all)

    f = 1  # Отображение тепловой карты
    if f == 1:
        all_kn, all_svm, all_dt, intervals = read_acc20(path + '\\split', initial_params['number_of_folders'])
        show_temp(all_kn, all_svm, all_dt, intervals, initial_params['number_of_folders'])

"""
     сценарий для обработки 1K изображений
"""
def scenario2 (initial_params):
    path = initial_params['path']

    # Подаются пути к данным, создаются txt файлы с psd. Использовать для перерасчёта.
    # data_to_features(initial_params['all_images'], initial_params['size_of_sample'], initial_params['p'],
    #             path + '\\true', path + '\\false', path + '\\split1K')

    # Классификация по имеющимся txt файлам
    interval = [[0, initial_params['count_of_features']]]
    classification(path + '\\split1K', initial_params['number_of_samples'], interval)

"""
     Косинусное преобразование
"""
def scenario3 (initial_params):
    show_img("E:\\NIRS\\Frequency\\Faces-HQ\\Flickr-Faces-HQ_10K\\1.jpg", False)
    cosinus_trans("E:\\NIRS\\Frequency\\Faces-HQ\\Flickr-Faces-HQ_10K\\1.jpg")