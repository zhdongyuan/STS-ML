import scipy.io
import os
import numpy as np
from scipy.stats import zscore

extrac_path = 'HBUED/Dataset/DE/'
save_path = 'HBUED/Dataset/data_independent/DE/time/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"路径不存在，已创建：{save_path}")
else:
    print(f"路径已存在：{save_path}")

dir_list = [f for f in os.listdir(extrac_path) if 'py' not in f]

i = 1
for test in dir_list:

    S = scipy.io.loadmat(extrac_path + test)

    DE = np.lib.stride_tricks.sliding_window_view(
        S['DE'], window_shape=(10, S['DE'].shape[1], S['DE'].shape[2]), axis=(0, 1, 2)
    ).squeeze()
    V = np.lib.stride_tricks.sliding_window_view(
        S['label_v'], window_shape=(S['label_v'].shape[0], 10), axis=(0, 1)
    ).squeeze(2)
    A = np.lib.stride_tricks.sliding_window_view(
        S['label_a'], window_shape=(S['label_a'].shape[0], 10), axis=(0, 1)
    ).squeeze(2)

    V_all_same = np.array([np.all(V[0, b, :] == V[0, b, 0])
                          for b in range(V.shape[1])])
    A_all_same = np.array([np.all(A[0, b, :] == A[0, b, 0])
                          for b in range(A.shape[1])])
    keep_mask = V_all_same & A_all_same

    DE_test = zscore(DE[keep_mask].transpose(0, 2, 1, 3))
    label_v_test = V[:, keep_mask, :][0, :, 0]
    label_a_test = A[:, keep_mask, :][0, :, 0]

    DE_train = None
    label_v_train = None
    label_a_train = None

    dir_list_train = [item for item in dir_list if item != test]

    for train in dir_list_train:
        data = scipy.io.loadmat(extrac_path + train)

        DE = np.lib.stride_tricks.sliding_window_view(
            data['DE'], window_shape=(10, data['DE'].shape[1], data['DE'].shape[2]), axis=(0, 1, 2)
        ).squeeze()
        V = np.lib.stride_tricks.sliding_window_view(
            data['label_v'], window_shape=(data['label_v'].shape[0], 10), axis=(0, 1)
        ).squeeze(2)
        A = np.lib.stride_tricks.sliding_window_view(
            data['label_a'], window_shape=(data['label_a'].shape[0], 10), axis=(0, 1)
        ).squeeze(2)

        V_all_same = np.array([np.all(V[0, b, :] == V[0, b, 0])
                               for b in range(V.shape[1])])
        A_all_same = np.array([np.all(A[0, b, :] == A[0, b, 0])
                               for b in range(A.shape[1])])
        keep_mask = V_all_same & A_all_same

        if DE_train is None:
            DE_train = zscore(DE[keep_mask].transpose(0, 2, 1, 3))
        else:
            DE_train = np.concatenate(
                [DE_train, zscore(DE[keep_mask].transpose(0, 2, 1, 3))], axis=0)

        if label_v_train is None:
            label_v_train = V[:, keep_mask, :][0, :, 0]
        else:
            label_v_train = np.concatenate(
                [label_v_train, V[:, keep_mask, :][0, :, 0]], axis=0)

        if label_a_train is None:
            label_a_train = A[:, keep_mask, :][0, :, 0]
        else:
            label_a_train = np.concatenate(
                [label_a_train, A[:, keep_mask, :][0, :, 0]], axis=0)

    np.save(f"{save_path}test_dataset_{i}.npy", DE_test)
    np.save(f"{save_path}test_labelset_{i}_v.npy", label_v_test)
    np.save(f"{save_path}test_labelset_{i}_a.npy", label_a_test)
    np.save(f"{save_path}train_dataset_{i}.npy", DE_train)
    np.save(f"{save_path}train_labelset_{i}_v.npy", label_v_train)
    np.save(f"{save_path}train_labelset_{i}_a.npy", label_a_train)

    print(f"save person {i}!")
    i += 1
