import os           # файлы
import re           # Регуляоные выражения
import glob         # Спиок файлов
import numpy as np
from scipy import ndimage   # Азимутальное среднее
from skimage import color   # Отображение изображений
from pprint import pprint   # Подключили Pprint для красоты выдачи текста
import matplotlib.pyplot as plt                 # Графики
from skimage.io import imread, imshow, show     # Отображение изображений
from scipy.fft import fft2, fftfreq, fftshift   # Преобразование фурье
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
def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
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


def azimuthalAverageBins(image, azbins, symmetric=None, center=None, **kwargs):
    """ Compute the azimuthal average over a limited range of angles
    kwargs are passed to azimuthalAverage """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2 * np.pi
    theta_deg = theta * 180.0 / np.pi

    if isinstance(azbins, np.ndarray):
        pass
    elif isinstance(azbins, int):
        if symmetric == 2:
            azbins = np.linspace(0, 90, azbins)
            theta_deg = theta_deg % 90
        elif symmetric == 1:
            azbins = np.linspace(0, 180, azbins)
            theta_deg = theta_deg % 180
        elif azbins == 1:
            return azbins, azimuthalAverage(image, center=center, returnradii=True, **kwargs)
        else:
            azbins = np.linspace(0, 359.9999999999999, azbins)
    else:
        raise ValueError("azbins must be an ndarray or an integer")

    azavlist = []
    for blow, bhigh in zip(azbins[:-1], azbins[1:]):
        mask = (theta_deg > (blow % 360)) * (theta_deg < (bhigh % 360))
        rr, zz = azimuthalAverage(image, center=center, mask=mask, returnradii=True, **kwargs)
        azavlist.append(zz)

    return azbins, rr, azavlist


def radialAverage(image, center=None, stddev=False, returnAz=False, return_naz=False,
                  binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None,
                  mask=None, symmetric=None):
    """
    Calculate the radially averaged azimuthal profile.
    (this code has not been optimized; it could be speed boosted by ~20x)
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the radial standard deviation instead of the average
    returnAz - if specified, return (azimuthArray,azimuthal_profile)
    return_naz   - if specified, return number of pixels per azimuth *and* azimuth
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and azimuthal
        profile so you can plot a step-form azimuthal profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2 * np.pi
    theta_deg = theta * 180.0 / np.pi
    maxangle = 360

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        # mask is only used in a flat context
        mask = np.ones(image.shape, dtype='bool').ravel()
    elif len(mask.shape) > 1:
        mask = mask.ravel()

    # allow for symmetries
    if symmetric == 2:
        theta_deg = theta_deg % 90
        maxangle = 90
    elif symmetric == 1:
        theta_deg = theta_deg % 180
        maxangle = 180

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(maxangle / binsize))
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # Find out which azimuthal bin each point in the map belongs to
    whichbin = np.digitize(theta_deg.flat, bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # azimuthal_prof.shape = bin_centers.shape
    if stddev:
        azimuthal_prof = np.array([image.flat[mask * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
        azimuthal_prof = np.array(
            [(image * weights).flat[mask * (whichbin == b)].sum() / weights.flat[mask * (whichbin == b)].sum() for b in
             range(1, nbins + 1)])

    # import pdb; pdb.set_trace()

    if interpnan:
        azimuthal_prof = np.interp(bin_centers,
                                   bin_centers[azimuthal_prof == azimuthal_prof],
                                   azimuthal_prof[azimuthal_prof == azimuthal_prof],
                                   left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(azimuthal_prof, azimuthal_prof)).ravel()
        return xarr, yarr
    elif returnAz:
        return bin_centers, azimuthal_prof
    elif return_naz:
        return nr, bin_centers, azimuthal_prof
    else:
        return azimuthal_prof


def radialAverageBins(image, radbins, corners=True, center=None, **kwargs):
    """ Compute the radial average over a limited range of radii """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])

    if isinstance(radbins, np.ndarray):
        pass
    elif isinstance(radbins, int):
        if radbins == 1:
            return radbins, radialAverage(image, center=center, returnAz=True, **kwargs)
        elif corners:
            radbins = np.linspace(0, r.max(), radbins)
        else:
            radbins = np.linspace(0, np.max(np.abs(np.array([x - center[0], y - center[1]]))), radbins)
    else:
        raise ValueError("radbins must be an ndarray or an integer")

    radavlist = []
    for blow, bhigh in zip(radbins[:-1], radbins[1:]):
        mask = (r < bhigh) * (r > blow)
        az, zz = radialAverage(image, center=center, mask=mask, returnAz=True, **kwargs)
        radavlist.append(zz)

    return radbins, az, radavlist


"""
    Азимутальное усреднение не учитывает углов 
"""
def getPSD1D(psd2D):
    h = psd2D.shape[0]  # Высота
    w = psd2D.shape[1]  # Ширина
    wc = w // 2         # Половина ширины
    hc = h // 2         # Половина высоты

    # создать массив целочисленных радиальных расстояний от центра
    Y, X = np.ogrid[0:h, 0:w]

    r = np.hypot(X - wc, Y - hc).astype(np.int) # Находим радиус окружности

    # Mean all psd2D pixels with label 'r' for 0 <= r <= wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))
    return psd1D


def GetPSD1D2(psd2D):
    h = psd2D.shape[0]  # Высота
    w = psd2D.shape[1]  # Ширина
    wc = w // 2         # Половина ширины
    hc = h // 2         # Половина высоты
    diag = np.hypot(wc, hc).astype(np.int) # Диагональ от центра

    # создать массив целочисленных радиальных расстояний от центра до угла
    Y, X = np.ogrid[0:h, 0:w]

    r = np.hypot(X - wc, Y - hc).astype(np.int) # Находим радиус окружности

    # Mean all psd2D pixels with label 'r' for 0 <= r <= wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, diag))
    return psd1D

"""
 Вычисляет psd1D. 
    Вход: изображения 
    Выход: psd1D (массив признаков)
"""
def calculations(img_nogrey, isavg):
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
    fft2 = np.fft.fftshift(np.log(1 + np.abs(fft2)))  # 1 + чтоб значения были от 0.Модуль перевод из комплексного
    #fft2 = np.fft.fftshift(1 + np.abs(fft2))  # Хуже работает

    # Добавить возможность деления на сумму усреднения
    if isavg == True:
        fft2 = fft2/sum(fft2, fft2[0])
    #psd1D = GetPSD1D2(fft2) #Обрубает углы
    psd1D = azimuthalAverage(fft2, binsize=1)

    return img, img_grey, fft2, psd1D

""" 
 Рисует Спектрограмму и Азимутальное усреднение для входящего изображения 
    Вход: изображение
    Выход: отсутствует 
"""
def show_img(img_nogrey, isavg):
    img, img_grey, fft2, psd1D = calculations(img_nogrey, isavg)

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
 Создаёт список имен изображений для train и test выборок
    Вход: режим
    Выход: отсутствует 
"""
def data_split(mode):   # mode: 20 or 40
    if mode == 20:
        truth = ['1_' + str(i) for i in range(10000)]          # 100KFake_10K               0 < i < 9999
        fake = ['0_' + str(i) for i in range(10000, 20000)]    # celebA-HQ_10K              10 000 < i < 19 999
        all = truth + fake

        y = [0 for i in range(20000)]
        for i in range(len(y)):
            if i <= 9999:
                y[i] = 1
        a,b = train_test_split(y, train_size=0.80, random_state=42)
        y = a + b

        all_train, all_test = train_test_split(all, train_size=0.80, random_state=42)   # stratify=y - равномерное распределение с 0 и 1
        # Массив для выборок train и test для 1K(тысячи)
        all_K1 = []
        for j in range(2):
            K1_train = [all_train[i] for i in range(0 + (j * 800), 800 + (j * 800))]
            K1_test = [all_test[i] for i in range(0 + (j * 200), 200 + (j * 200))]
            all_K1.append({0: K1_train, 1: K1_test})

    elif mode == 40:
        fake = ['1_' + str(i) for i in range(10000)]           # 100KFake_10K               0 < i < 9999
        celeba = ['0_' + str(i) for i in range(10000, 20000)]  # celebA-HQ_10K              10 000 < i < 19 999
        flickr = ['0_' + str(i) for i in range(20000, 30000)]  # Flickr-Faces-HQ_10K        20 000 < i < 29 999
        thispe = ['1_' + str(i) for i in range(30000, 40000)]  # thispersondoesntexists_10K 30 000 < i < 39 999
        all = fake + celeba + flickr + thispe

        # y = [0 for i in range(40000)]
        # for i in range(len(y)):
        #     if i <= 29999:
        #         y[i] = 1
        # a, b = train_test_split(y, train_size=0.80, random_state=42)
        # y = a + b

        all_train, all_test = train_test_split(all, train_size=0.80, random_state=42) # stratify=y

        # Массив для выборок train и test для каждой 1K(тысячи)
        all_K1 = []
        for j in range(40):
            K1_train = [all_train[i] for i in range(0 + (j*800), 800 + (j*800))]
            K1_test = [all_test[i] for i in range(0 + (j*200), 200 + (j*200))]
            all_K1.append( {0:K1_train, 1:K1_test} )           # 0-train 1-test

    return all_K1

"""
 Находит путь к изображению по номеру
    Вход: номер изображения (имя изображения)
    Выход: путь до изображения
"""
def find_path(number):
    path = "E:\\NIRS\\Frequency\\Faces-HQ\\"

    if number <= 9999:
        path += '100KFake_10K'+'\\'+str(number)+'.jpg'
    elif number <= 19999:
        path += 'celebA-HQ_10K'+'\\'+str(number - 10000)+'.jpg'
    else:
        if number <= 29999:
            path += 'Flickr-Faces-HQ_10K'+'\\'+str(number - 20000)+'.jpg'
        elif number <= 39999:
            path += 'thispersondoesntexists_10K'+'\\'+str(number - 30000)+'.jpg'

    return path

"""
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
            img, img_grey, fft2, psd1D = calculations(path, False)
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
            img, img_grey, fft2, psd1D = calculations(path, False)
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
    Вход: путь до папки 
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


if __name__ == '__main__':
    # Начало
    start_time = datetime.now()

    isavg = False
    # isavg = input('Усреднение: ')

    path = "E:\\NIRS\\Frequency\\Faces-HQ\\celebA-HQ_10K"
    images = os.listdir(path)

    a = 0   # Перещёт значений psd
    if a == 1:
        list_allK1 = data_split(40)
        x_train, y_train, x_test, y_test = list2psD1(list_allK1, 'E:\\NIRS\\Frequency\\Faces-HQ\\split\\all')

    path_0 = "E:\\NIRS\\Frequency\\Faces-HQ\\split\\all"

    all_kn = []
    all_svm = []
    all_dt = []
    intervals = []

    # b = 1   # Перещёт классификаторами для интервала в 10 признаков
    # if b == 1:
    #     for j in range(40):
    #         all_kn = []
    #         all_svm = []
    #         all_dt = []
    #         intervals = []
    #         for i in range(0, 720, 10):
    #             if i == 710:
    #                 interval = [[i, i+14]]
    #             else:
    #                 interval = [[i, i+10]]
    #
    #             kn, svm, dt = classifier(path_0 +'\\'+ str(j), interval)
    #             all_kn.append(kn)
    #             all_svm.append(svm)
    #             all_dt.append(dt)
    #             intervals.append(interval[0])
    #
    #         save_in_1K(path_0 +'\\'+ str(j) + '\\acc.txt', all_kn, all_svm, all_dt, intervals)

    b = 0  # Перещёт классификаторами для интервала в 20 признаков из участков по 10 из разных частей.
    if b == 1:
        for j in range(0, 40):
            all_kn = []
            all_svm = []
            all_dt = []
            intervals = []

            for i in range(0, 720, 10):
                for k in range(10+i, 720, 10):
                    interval = []
                    interval.append([i, i + 10])

                    if k == 710:
                        interval.append([k, k + 14])
                    else:
                        interval.append([k, k + 10])

                    kn, svm, dt = classifier(path_0 + '\\' + str(j), interval)
                    all_kn.append(kn)
                    all_svm.append(svm)
                    all_dt.append(dt)
                    intervals.append(interval[0])
                    intervals.append(interval[1])
                print(f'Выборка:{j},Интервал:{i}')

            save_in_1K(path_0 + '\\' + str(j) + '\\acc20.txt', all_kn, all_svm, all_dt, intervals)


    c = 0   # Отображение данных классификаторов
    if c == 1:
        for i in range(40):
            kn_all, svm_all, dt_all = read_acc(path_0 + '\\' +str(i)+ '\\acc.txt')
            show_acc(len(kn_all), kn_all, svm_all, dt_all)

    all_kn = []
    all_svm = []
    all_dt = []
    intervals = []

    d = 0  # Отображение тепловой карты
    if d == 1:
        all_kn, all_svm, all_dt, intervals = read_acc20(path_0, 40)
        show_temp(all_kn, all_svm, all_dt, intervals, 40)

    #show_img(path + '\\12.jpg', isavg)
    show_img(path + 'E:\\NIRS\\Frequency\\Faces-HQ2\\true\\images1024x1024-20191222T221133Z-001\\images1024x1024\\59000', isavg)
    # classifier_1k(mode=0)

    # Конец
    print(datetime.now() - start_time)
