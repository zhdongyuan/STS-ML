import scipy.io
import os
import numpy as np

extrac_path = 'HBUED/Dataset/feature/'
save_path = 'HBUED/Dataset/DE/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"路径不存在，已创建：{save_path}")
else:
    print(f"路径已存在：{save_path}")

dir_list = os.listdir(extrac_path)

# label = scipy.io.loadmat(label_path)
# label = label['label'][0]

for f in dir_list:
    if 'mat' not in f:
        continue

    S = scipy.io.loadmat(extrac_path + f)
    DE = S['data'].transpose(0, 2, 1)
    label_v = S['valence_labels']
    label_a = S['arousal_labels']

    mdic = {"DE": DE, "label_v": label_v, "label_a": label_a}

    scipy.io.savemat(save_path + f, mdic)
    print(extrac_path + f, '->', save_path + f)
