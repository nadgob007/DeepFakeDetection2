from scenarios import *

initial_params = {
    'all_images': 20000,                                        # колличество всех фотографий
    'size_of_sample': 1000,                                     # колличество фотографий в выборке (папке)
    'number_of_samples': 0,                                     # колличество выборок(папок) по size_of_sample фотографий
    'p': 0.80,                                                  # Процент тренировочной части выборки
    'count_of_features': 724,                                   # Общее количество признаков для 1 изображения
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"
}
initial_params['number_of_samples'] = int(initial_params['all_images'] / initial_params['size_of_sample'])

initial_params2 = {
    'all_images': 1000,                                         # колличество всех фотографий
    'size_of_sample': 1000,                                     # колличество фотографий в выборке (папке)
    'number_of_samples': 0,                                     # колличество выборок(папок) по size_of_sample фотографий
    'p': 0.80,                                                  # Процент тренировочной части выборки
    'count_of_features': 724,                                   # Общее количество признаков для 1 изображения
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"
}
initial_params2['number_of_samples'] = int(initial_params2['all_images'] / initial_params2['size_of_sample'])

if __name__ == '__main__':
    # Начало
    start_time = datetime.now()
    print('Start in:', start_time)

    scenario1(initial_params) # Для функции классификации дописать более полную точность

    # scenario2(initial_params2)

    # Конец
    print('End in:', datetime.now() - start_time)