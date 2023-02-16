import numpy
import os       # файлы
import glob     # Спиок файлов

from scipy.io import loadmat

path = 'G:\\data_set\\hmdb51_sta\\cartwheel\\Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_5.avi.tform.mat'

m = loadmat(path)
print(m)
mask_out = m["mask_out"]
print(mask_out.shape)
