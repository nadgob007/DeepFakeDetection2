import os           # файлы
import re           # Регуляоные выражения
import glob         # Спиок файлов
import numpy as np
from scipy import ndimage   # Азимутальное среднее
from skimage import color   # Отображение изображений
from pprint import pprint   # Подключили Pprint для красоты выдачи текста
import matplotlib.pyplot as plt                 # Графики
from matplotlib import gridspec
from skimage.io import imread, imshow, show     # Отображение изображений
from scipy.fft import fft2, fftfreq, fftshift, dct   # Преобразование фурье
from sklearn.model_selection import train_test_split    # Разбиение данных на обучение и тестирования
from sklearn.neighbors import KNeighborsClassifier      # Классификация ближайших соседей
from sklearn.pipeline import make_pipeline              # Классификация векторов поддержки С
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score     # Классификатор дерева решений
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime                           # Время выполнения скрипта
import time

"""
    Сторонние функции азимутального усреднения
"""
def azimuthalAverage(image, center=None, stddev=False, median=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
                     mask=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    :param median:

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape, dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    # nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat, bins)
        # This method is still very slow; is there a trick to do this with histograms?
        radial_prof = np.array([image.flat[mask.flat * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
        if median:
            w, h = r.shape
            med = [np.array([]) for i in range(maxbin)]
            for i in range(h-1):
                for j in range(w-1):
                    a = int(np.round(r[i][j]))
                    b = image[i][j]
                    med[a]=np.append(med[a], [b])

            radial_prof = np.empty((maxbin))

            for i in range(1, len(med)):
                a=med[i] #TODO: Что-то не так"
                radial_prof[i-1] = np.median(med[i])
        else:
            radial_prof = np.histogram(r, bins, weights=(image * weights * mask))[0] / \
                          np.histogram(r, bins, weights=(mask * weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers, bin_centers[radial_prof == radial_prof],
                                radial_prof[radial_prof == radial_prof], left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof

"""
 Вычисляет psd1D. 
    Вход: изображения 
    Выход: psd1D (массив признаков)
"""
# calculations -> calculate_features
def calculate_features(img_nogrey, isavg):
    try:
        img = imread(img_nogrey)  # Цветное изображение
    except:
        f = open('err.txt', 'a')
        f.write(img_nogrey)
        f.close()
        return 0, 0, [], []
    else:
        print('Исключений не произошло')

    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого

    # Быстрое преобразование Фурье FFT
    fft2 = np.fft.fft2(img_grey)  # Использование FFT

    # Перемещение картинки в центр и использование модуля. Спектрально-логарифмическое преобразование
    # 1 + чтоб значения были от 0.Модуль перевод из комплексного

    fft2 = np.fft.fftshift(np.log(1 + np.abs(fft2)))
    #fft2 = np.fft.fftshift(1 + np.abs(fft2))  # Хуже работает

    # Добавить возможность деления на сумму усреднения
    if isavg == True:
        fft2 = fft2/sum(fft2, fft2[0])
    psd1D = azimuthalAverage(fft2, binsize=1, median=True)

    return img, img_grey, fft2, psd1D

""" 
 Рисует Спектрограмму и Азимутальное усреднение для входящего изображения 
    Вход: изображение
    Выход: отсутствует 
"""
def show_img(img_nogrey, isavg):
    img, img_grey, fft2, psd1D = calculate_features(img_nogrey, isavg)

    # Простотранство для отображения
    fig = plt.figure(figsize=(15, 5))

    # Цветная. Значения от 0 до 255
    fig.add_subplot(2, 3, 1)
    plt.title("Изображение до обработки", fontsize=12)
    imshow(img)

    # В оттенках серого. Значения от 0 до 1
    fig.add_subplot(2, 3, 2)
    plt.title("Изображение в отенках серого", fontsize=12)
    imshow(img_grey)

    # Быстрое преобразование Фурье FFT. Значения
    fig.add_subplot(2, 3, 3)
    plt.ylabel('Амплитуда', fontsize=10)
    plt.xlabel('Частота', fontsize=10)
    plt.title("Спектрограмма", fontsize=12)
    imshow(fft2, cmap='gray')   # Отображать серым

    # Азимутальное усреднение
    fig.add_subplot(2, 1, 2)
    plt.plot(psd1D, color='green', linestyle='-', linewidth=1, markersize=4)
    plt.ylabel('Энергетический спектр', fontsize=10)
    plt.xlabel('пространственная частота', fontsize=10)
    plt.title("Азимутальное усреднение", fontsize=12)

    plt.tight_layout()
    show()

    return 0

"""
[-] устарела не используется
 Примнимает список имен всех выборок используемых изображений, вычисляет массив признаков, сохраняет файл с маасивом в 
 указанную папку, возвращает x_train/x_test - вектор признаков и y_train/y_test - вектор классов
    Вход: список имен используемых изображений, путь до папки сохранения
    Выход: массивы признаков и классов для train и test 
"""
def list2psD1(list_allK1, path_folder):

    for i in range(len(list_allK1)):

        # Обучающий набор
        list_train = list_allK1[i].get(0)
        train_numbers = []
        y_train = []
        x_train = []
        for img in list_train:
            truth = img.split("_")
            y_train.append(truth[0])
            train_numbers.append(truth[1])

        count = 0
        for j in train_numbers:
            path = find_path(int(j))
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
            truth = img.split("_")
            y_test.append(truth[0])
            test_numbers.append(truth[1])

        count = 0
        for j in test_numbers:
            path = find_path(int(j))
            img, img_grey, fft2, psd1D = calculate_features(path, False)
            x_test.append(psd1D)
            print(f'Test №{i}')
            print(f'Номер: {j}')
            print(f'Прогресс: {count/2}%')
            count += 1

        # Путь до файла сохранения
        psd_save(path_folder +'\\'+ str(i) + '\\test_psd.txt', x_test, y_test, test_numbers)

    return x_train, y_train, x_test, y_test

""" 
 Сохраняет массивы признаков и классов psd1D в текстовый файл и возвращает сколько строк сохранил
    Вход: путь до файла сохранения, массив признаков, массив классов, название файлов
    Выход: сколько строк сохранил
"""
def psd_save(path, x, y, numbers):

    # Формируем шапку таблицы в psd.txt
    f = open(path, 'w')
    form = 'Image[№]\t PSD_1D\n'
    f.write(form)
    line = ''
    elem = 0
    while elem != len(x):
        line += str(y[elem]) + '_' + numbers[elem] + '\t' + str([x[elem][i] for i in range(len(x[elem]))]) + '\n'
        elem += 1
    f.write(line)
    f.close()

    return elem

"""
 Читает файл c признаками и возвращает массив признаков и классов
    Вход: путь до файла чтения
    Выход: x - массив признаков, y - массив классов
"""
def read_save(path, interval):
    f = open(path, 'r')
    line = f.readline()
    x = []
    numbers = []
    while line:
        line = f.readline()
        if len(line) == 0:
           break
        result = re.split(r'\t', line)
        numbers.append(result.pop(0))
        result = result[0][1:-2]
        result = re.split(r', ', result)
        tmp = []
        for i in interval:
            for j in range(i[0], i[1]):
                if result[j] == '':
                    print(j)
                    # for c in range(i[0], i[1]):
                    #     tmp.append(float(0))
                    break
                else:
                    tmp.append(float(result[j]))
        x.append(tmp)

    y = []
    for i in numbers:
        truth = i.split("_", 1)
        y.append(truth[0])

    f.close()
    return x, y

"""
 Классифицирует полученные из файла массивы признаков по выборкам train и test
    Вход: путь до папки,
    Выход: точность для KN, SVM, DT 
"""
def classifier(path_folder, interval):
    x_train, y_train = read_save(path_folder + '\\train_psd.txt', interval)
    x_test, y_test = read_save(path_folder + '\\test_psd.txt', interval)

    count_train = 0
    for i in range(len(y_train)):
        if y_train[i] == '1':
            count_train += 1

    count_test = 0
    for i in range(len(y_test)):
        if y_test[i] == '1':
            count_test += 1

    # Классификация ближайших соседей
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)

    predicts_KN = neigh.predict(x_test)
    accuracy_KN = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_KN[i]:
            accuracy_KN +=1


    # Классификация векторов поддержки С радиальной базисной функции
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(x_train, y_train)

    predicts_SVM = clf.predict(x_test)
    accuracy_SVM = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_SVM[i]:
            accuracy_SVM += 1


    # Классификатор дерева решений
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)

    predicts_DT = clf.predict(x_test)
    accuracy_DT = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_DT[i]:
            accuracy_DT += 1

    return accuracy_KN, accuracy_SVM, accuracy_DT

"""
 Создает файла со статистикой точности
    Вход: путь до файла, точности для KN, SVM, DT.
    Выход: количество строк
"""
def accuracy_save(path, kn, svm, dt):
    # Формируем шапку таблицы в acc.txt
    f = open(path, 'w')
    form = '№\t KN(%)\t SVM(%)\t DT(%)\n'
    f.write(form)
    line = ''
    rows = 0
    for i in range(len(kn)):
        line += f'{i}\t {kn[i]/2}\t {svm[i]/2}\t {dt[i]/2}\n'
        f.write(line)
        line = ''
        rows = i
    f.close()
    return rows

"""
 Читает файл со статистикой по всем 40 выборкам. Возвращает массив точностей по каждой выборке для каждого классификатор
    Вход: путь до файла чтения
    Выход: массивы точностей для KN, SVM, DT
"""
def read_acc(path):
    f = open(path, 'r')
    line = f.readline() # Игнорируем шапку
    all_kn = []
    all_svm = []
    all_dt = []
    while line:
        line = f.readline()
        if len(line) == 0:
            break
        result = re.split(r'\t ', line)
        all_kn.append(float(result[1]))
        all_svm.append(float(result[2]))
        all_dt.append(float(result[3]))
    f.close()

    return all_kn, all_svm, all_dt

"""
[+]
 Читает файл со статистикой по всем 40 выборкам. Возвращает массив точностей по каждой выборке для каждого классификатора
    Вход: путь до файла чтения, колличество папок
    Выход: массивы точностей для KN, SVM, DT и интервал признаков (сколько признаков)
"""
def read_acc20(path, number_of_folders):
    all_kn = []
    all_svm = []
    all_dt = []
    intervals = []
    for i in range(number_of_folders):
        f = open(path + f'\\{i}\\acc20.txt', 'r')
        line = f.readline()  # Игнорируем шапку
        kn = []
        svm = []
        dt = []
        while line:
            line = f.readline()
            if len(line) == 0:
                break
            result = re.split(r'\t ', line)
            if i == 0:
                intervals.append(result[0])
            kn.append(float(result[1]))
            svm.append(float(result[2]))
            dt.append(float(result[3]))
        f.close()
        all_kn.append(kn)
        all_svm.append(svm)
        all_dt.append(dt)
        print(i)

    return all_kn, all_svm, all_dt, intervals

"""
[+]
 Строит тепловые карты для каждого массива точностей 
    Вход: массивы точностей для KN, SVM, DT и интервал признаков (сколько признаков), колличество папок
    Выход: отсутствует
"""
def show_temp(all_kn, all_svm, all_dt, intervals, number_of_folders):
    kn = []
    svm = []
    dt = []
    for j in range(len(all_kn[0])):
        mean_kn = []
        mean_svm = []
        mean_dt = []
        for i in range(number_of_folders):
            mean_kn.append(all_kn[i][j])
            mean_svm.append(all_svm[i][j])
            mean_dt.append(all_dt[i][j])
        kn.append(np.mean(mean_kn))
        svm.append(np.mean(mean_svm))
        dt.append(np.mean(mean_dt))

    a = [[] for i in range(71)]
    b = [[] for i in range(71)]
    c = [[] for i in range(71)]

    count = 0
    for i in range(72):
        count +=i

    k = 0
    min_a = 100
    min_b = 100
    min_c = 100
    max_a = 0
    max_b = 0
    max_c = 0
    for j in range(71):
        for i in range(71):
            if i < j:
                a[j].append(0)
                b[j].append(0)
                c[j].append(0)
            else:
                break

        for i in range(j, 71):
            a[j].insert(i, int(kn[k]))
            b[j].insert(i, int(svm[k]))
            c[j].insert(i, int(dt[k]))

            if kn[k] < min_a:
                min_a = kn[k]
            if svm[k] < min_b:
                min_b = svm[k]
            if dt[k] < min_c:
                min_c = dt[k]

            if kn[k] > max_a:
                max_a = kn[k]
            if svm[k] > max_b:
                max_b = svm[k]
            if dt[k] > max_c:
                max_c = dt[k]
            i+=1
            k+=1
        print(j)


    fig = plt.figure(figsize=(15, 5))

    # KN
    fig.add_subplot(1, 3, 1)
    plt.title(f"Kn (min/max)\n{min_a}-{max_a}", fontsize=12)
    plt.matshow(a, 0)
    fig.colorbar(plt.matshow(a, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    # SVM
    fig.add_subplot(1, 3, 2)
    plt.title(f"SVM (min/max)\n{min_b}-{max_b}", fontsize=12)
    plt.matshow(b, 0)
    fig.colorbar(plt.matshow(b, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    # DT
    fig.add_subplot(1, 3, 3)
    plt.title(f"DT (min/max)\n{min_c}-{max_c}", fontsize=12)
    plt.matshow(c, 0)
    fig.colorbar(plt.matshow(c, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    plt.show()

"""
 Отображает график для общей статистики точности каждого классификатора
    Вход: колличество выборок, массивы точностей для KN, SVM, DT
    Выход: отсутствует
"""
def show_acc(num, all_kn, all_svm, all_dt):
    #  Задаем смещение равное половине ширины прямоугольника:
    x1 = np.arange(0, num) - 0.3
    x2 = np.arange(0, num) + 0
    x3 = np.arange(0, num) + 0.3

    mins = [min(all_kn), min(all_svm), min(all_dt)]
    print(f'Наименьшее:{mins}')

    y1 = [all_kn[i] for i in range(len(all_kn))]
    y2 = [all_svm[i] for i in range(len(all_svm))]
    y3 = [all_dt[i] for i in range(len(all_dt))]

    #y_masked = np.ma.masked_where(int(y1) < 50, y1)

    fig, ax = plt.subplots()
    plt.ylim(min(mins), 100)

    ax.bar(x1, y1, width=0.2, label='KN')
    ax.bar(x2, y2, width=0.2, label='SVM', color='orange')
    ax.bar(x3, y3, width=0.2, label='DT' ,color='green')

    ax.legend(loc = "upper left")

    ax.set_title(f'Точность KN, SVM, DT')
    ax.set_facecolor('seashell')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure
    fig.set_facecolor('floralwhite')

    plt.show()

    return 0

"""
 Классифицирует 1ну выборку в 1ой папке и сохраняет файл с точностямим 
    Вход: режим
    Выход: отсутствует
"""
def classifier_1k(mode=1):
    path = 'E:\\NIRS\\Frequency\\Faces-HQ\\split\\100KFake_10K+celebA-HQ_10K'
    if mode == 1:
        list_allK1 = data_split(20)
        x_train, y_train, x_test, y_test = list2psD1(list_allK1, path)

    for i in range(2):
        kn, svm, dt = classifier(path + '\\' + str(i))
        all_kn.append(kn)
        all_svm.append(svm)
        all_dt.append(dt)
    accuracy_save(path + '\\acc.txt', all_kn, all_svm, all_dt)

    print(f'KN, SVM, DT: {kn/2}%, {svm/2}%, {dt/2}%')

    return 0

"""
[+]
 Сохраняет значения классификаторов для 10 или 20 признаков 
    Вход: путь до вайла сохранения, массивы точностей для KN, SVM, DT, интервалы признаков, режим 10 или 20
    Выход: сохраненные строки
"""
def save_in_1K(path, kn, svm, dt, intervals, mode):
    # Формируем шапку таблицы в acc[№].txt
    f = open(path, 'w')
    form = 'Interval\t KN(%)\t SVM(%)\t DT(%)\n'
    f.write(form)
    line = ''
    rows = 0
    j = 0
    for i in range(len(kn)):
        if mode == 10:
            line += f'{intervals[j][0]}-{intervals[j][1]}\t {kn[i] / 2}\t {svm[i] / 2}\t {dt[i] / 2}\n'
        elif mode == 20:
            line += f'{intervals[j][0]}-{intervals[j][1]}:{intervals[j + 1][0]}-{intervals[j + 1][1]}\t {kn[i] / 2}\t {svm[i] / 2}\t {dt[i] / 2}\n'
        f.write(line)
        line = ''
        rows = i
        if mode == 10:
            j += 1
        elif mode == 20:
            j += 2
    f.close()
    return rows


"""
 Составляет массивы путей до настоящих и поддельных изображений. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
    Вход: 
        n - размер выборки, 
        path_true и path_false - пути до папок с настоящими и поддельными изображениями
    Выход: половины от массивов путей до настоящих и поддельных изображений.
"""
def get_data_list(n, path_true, path_false):

    true_items = []
    for dirpath, dirnames, filenames in os.walk(path_true):
        if not (len(filenames) == 0) and len(dirnames) == 0:
            for i in filenames:
                true_items.append(dirpath+'\\'+i)

    false_items = []
    for dirpath, dirnames, filenames in os.walk(path_false):
        if not (len(filenames) == 0) and len(dirnames) == 0:
            for i in filenames:
                false_items.append(dirpath + '\\' + i)

    return true_items[0:int(n/2)], false_items[0:int(n/2)]

"""
 Примнимает список имен используемых изображений, вычисляет массивы признаков и записывает в файл
    Вход: 
        list_allK1 - массив объектов содержащих, имена путей до изображений и обозначения (0 или 1) 
        path_folder - путь до папки split (в которой будут лежать папки с выборками)
    Выход: x_train/x_test - вектора признаков, y_train/y_test - вектора классов
"""
def list2psD1_2(list_allK1, path_folder, features):

    for i in range(0, len(list_allK1)):
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
            img, img_grey, fft2, psd1D = features(path, False)
            x_train.append(psd1D)
            print(f'Train №{i}')
            print(f'Номер: {j}')
            print(f'Прогресс: {count/8}%')
            count += 1

        # Создаем папку с номером набора 1К, если её нет
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
            img, img_grey, fft2, psd1D = features(path, False)
            x_test.append(psd1D)
            print(f'Test №{i}')
            print(f'Номер: {j}')
            print(f'Прогресс: {count/2}%')
            count += 1

        # Путь до файла сохранения
        psd_save(path_folder +'\\'+ str(i) + '\\test_psd.txt', x_test, y_test, test_numbers)

    return x_train, y_train, x_test, y_test

"""
 1. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
 2. помечаем 1 - true, 0 - false
 3. соединяем и перемешиваем.
 4. разбиваем массив на n/sample папок (20)
 5. Получаем массивы признаков и классов для изображений
    Вход: 
        n - размер выборки
        sample - размер 1ой выборки
        tf - train test соотношение
        path_true, path_false - пути до папок true false
        path - куда сохранить psd
    Выход: train.txt и test.txt файлы с путями до *.png файлов
"""
def data_to_psd(n, sample, tf, path_true, path_false, path, features = calculate_features):

    # 1. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
    true, false = get_datasets_list(n, path_true)

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
    x_train, y_train, x_test, y_test = list2psD1_2(all_K1, path, features)
    return 0


""" Сделать из этого обобщенную функцию с заменяемыми методами"""
def data_to_features(n, sample, tf, path_true, path_false, path, features = calculate_features):

    # 1. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
    true, false = get_datasets_list(n, path_true)

    # 2. помечаем 1 - true, 0 - false
    true = [[1, true[i]] for i in range(len(true))]
    false = [[0, false[i]] for i in range(len(false))]

    # 3. соединяем и перемешиваем.
    true_false = true + false
    all_train, all_test = train_test_split(true_false, train_size=tf, random_state=42)

    # 4. разбиваем массив на n/sample папок
    all_K1 = []
    for j in range(int(n/sample)):
        K1_train = [all_train[i] for i in range(0 + (j * int(sample*tf)), int(sample*tf) + (j * int(sample*tf)))]
        K1_test = [all_test[i] for i in range(0 + (j * int(sample-sample*tf)), int(sample-sample*tf) + (j * int(sample-sample*tf)))]
        all_K1.append({0: K1_train, 1: K1_test})  # 0-train 1-test

    # 5. Получаем массив признаков для изображений
    x_train, y_train, x_test, y_test = list2psD1_2(all_K1, path, features)
    return 0

"""
[? зачем принт]
 Классифицирует все выборки, сохраняет точности по каждой выборке и выводит в консоль 
    Вход: 
        path - куда сохранить psd
        number_folders - количество выборок(папок)
        interval - колличество признаков которое будет использоваться при классификации
    Выход:
"""
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

"""
 Классифицирует указанное количество выборок используя 10 признаков, сохраняет точности по каждой выборке
    Вход:
        path - куда сохранить psd
        number_folders - количество выборок(папок)
    Выход:
"""
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
        save_in_1K(path +'\\'+ str(j) + '\\acc10.txt', all_kn, all_svm, all_dt, intervals, mode=10)

"""
 Классифицирует указанное количество выборок используя 20 признаков, сохраняет точности по каждой выборке
    Вход:
        path - куда сохранить psd
        number_folders - количество выборок(папок)
    Выход:
"""
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


""" ___________________________________________
                    Сценарий 3
    ___________________________________________
"""

"""
 Вычисление косинусного преобразования, μ – среднего по выборке и σ - среднеквадратического отклонения.
    Вход:
        img_nogrey - изображение
        is_zigzag - распологать ли элементы в зиг-заг
    Выход:
"""
def cosinus_trans(img_nogrey, is_zigzag=True):
    img = imread(img_nogrey)
    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого

    w, h = img_grey.shape
    f = 8
    count = int(w/f)

    blocks = []
    blocks_dct = []
    averages = [[] for i in range(f*f)]

    for i in range(count):
        for j in range(count):
            blocks.append(img_grey[i*f:(i*f+f), j*f:(j*f+f)])

            # Косинусное преобразование
            block = dct(img_grey[i*f:(i*f+f), j*f:(j*f+f)])
            # модуль от косинусного перобразования надо ли ?
            #block = np.abs(block)
            if is_zigzag:
                block = zigzag(block)
                # Создать 2д массив из 1д
                #block = np.reshape(block, (-1, 8))
                for i in range(len(block)):
                    averages[i].append(block[i])
            blocks_dct.append(block)

    averages_m = [np.mean(i) for i in averages]
    averages_beta = [np.std(i)/2**(1/2) for i in averages]

    return averages_beta

"""
 Демонстрация косинусного преобразования
    Вход:
        img_nogrey - изображение
        is_zigzag - распологать ли элементы в зиг-заг
    Выход: нет
"""
def cosinus_trans_show(img_nogrey, is_zigzag=True):
    img = imread(img_nogrey)
    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого
    w, h = img_grey.shape
    f = 8
    count = int(w/f)
    blocks = []
    blocks_dct = []
    averages = [[] for i in range(f*f)]

    for i in range(count):
        for j in range(count):
            blocks.append(img_grey[i*f:(i*f+f), j*f:(j*f+f)])

            # Косинусное преобразование
            block = dct(img_grey[i*f:(i*f+f), j*f:(j*f+f)])

            # модуль от косинусного перобразования надо ли ?
            #block = np.abs(block)
            if is_zigzag:
                block = zigzag(block)

                # Создать 2д массив из 1д
                #block = np.reshape(block, (-1, 8))
                for i in range(len(block)):
                    averages[i].append(block[i])
            blocks_dct.append(block)

    averages_m = [np.mean(i) for i in averages]
    averages_beta = [np.std(i)/2**(1/2) for i in averages]

    # соединяем блоки в 1 изображение
    c = []
    for i in range(0, 128):
        b = c

        a = blocks_dct[i*128]
        for j in range(1, 128):
           a = np.hstack([a, blocks_dct[i*128 + j] ])

        if i == 0:
            c = a.copy()
        else:
            c = np.vstack([b, a])

    fig = plt.figure(figsize=(8, 8))
    imshow(c, cmap='gray')

    dct2 = np.log(1 +np.abs(dct(img_grey)))
    # Простотранство для отображения
    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(2, 3, 1)
    plt.title("Изображение до обработки", fontsize=12)
    imshow(img)

    # В оттенках серого. Значения от 0 до 1
    fig.add_subplot(2, 3, 2)
    plt.title("Изображение в отенках серого", fontsize=12)
    imshow(img_grey)

    fig.add_subplot(2, 3, 3)
    imshow(dct2, cmap='gray')  # Отображать серым
    show()

"""
  Зиг-загом переписывает 2d массив в 1d массив. 
    Вход: массив 
    Выход: массив в зиг-заг развёртке
"""
def zigzag(matrix):
    zigzag = []
    for index in range(1, len(matrix) + 1):
        slice = [i[:index] for i in matrix[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag

    for index in range(1, len(matrix)):
        slice = [i[index:] for i in matrix[index:]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag

"""
 Составляет массивы путей до настоящих и поддельных изображений. получаем массивы путей до файлов картинок.
    Вход: 
        n - размер выборки, 
        path_true и path_false - пути до папок с настоящими и поддельными изображениями
    Выход: массивов путей до настоящих и поддельных изображений.
"""
def get_datasets_paths(path_true, path_false):
    true_datasets = [[os.path.join(path_true, dirpath)] for dirpath in os.listdir(path_true)]
    for j in true_datasets:
        true_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    true_items.append(dirpath +'\\'+ i)
        j.append(true_items)

    false_datasets = [[os.path.join(path_false, dirpath)] for dirpath in os.listdir(path_false)]
    for j in false_datasets:
        false_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    false_items.append(dirpath +'\\'+ i)
        j.append(false_items)

    return true_datasets, false_datasets

""" 
 
    Вход: путь до файла сохранения, название сохраняемого, массив бета
    Выход: нет
"""
def dct_save(path, name, averages_beta):
    # Сохраняем 1 массив созданый по 1 изображению
    f = open(path + '\\' + 'dct.txt', 'a')
    line = ''
    line += str(name) +'\t'+ str([i for i in averages_beta]) +'\n'
    f.write(line)
    f.close()

"""
 
    Вход: путь до файла чтения
    Выход: averages - массив матриц beta
"""
def dct_read(path):
    f = open(path +'\\'+ 'dct.txt', 'r')
    line = '.'
    averages = []
    names = []
    while line:
        line = f.readline()
        if len(line) == 0:
           break
        result = re.split(r'\t', line)
        names.append(result.pop(0))
        result = result[0][1:-2]
        result = re.split(r', ', result)
        tmp = [float(i) for i in result]
        averages.append(tmp)
    f.close()
    return averages

def data_to_frequencies(path_true, path_false, path):
    # 1. получаем массив путей до файлов картинок.
    true, false = get_datasets_paths(path_true, path_false)

    # 2. Высчитываем и сохраняем матрицу(64) для каждого изображения в каждом датасете
    # TODO: (объединить этот этап в 1ну функцию и оба фора объединить)!!!!
    for dataset in true:
        averages = [[] for i in range(len(dataset[1]))]

        # Путь до папки датасета
        path_folder = path +'\\true\\'+ os.path.basename(os.path.normpath(dataset[0]))
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        bookmark = len(dct_read(path_folder))
        if len(dataset[1]) > bookmark:
            count = bookmark
            dataset_c = dataset[1][bookmark:]
            for i in dataset_c:
                averages_beta = cosinus_trans(i)
                dct_save(path_folder, i, averages_beta)
                for j in range(len(averages_beta)):
                    averages[j].append(averages_beta[j])
                print(count, '/', len(dataset[1]))
                count+=1
        beta = [np.mean(j) for j in averages]

    for dataset in false:
        averages = [[] for i in range(len(dataset[1]))]

        # Путь до папки датасета
        path_folder = path + '\\false\\' + os.path.basename(os.path.normpath(dataset[0]))
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        bookmark = len(dct_read(path_folder))
        if len(dataset[1]) > bookmark:
            count = bookmark
            dataset_c = dataset[1][bookmark:]
            for i in dataset_c:
                averages_beta = cosinus_trans(i)
                dct_save(path_folder, i, averages_beta)
                for j in range(len(averages_beta)):
                    averages[j].append(averages_beta[j])
                print(count, '/', len(dataset[1]))
                count += 1
        beta = [np.mean(j) for j in averages]
