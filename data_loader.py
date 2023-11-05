import numpy as np
import os, pickle, torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler

def label_selection(label,label_type='A',num_class=2):
    if label_type == 'A':
        label = label[:, 1]
    elif label_type == 'V':
        label = label[:, 0]
    elif label_type == 'VA':
        label = label[:, 0:2]
    if num_class == 2:
        label = np.where(label <= 5, 0, label)
        label = np.where(label > 5, 1, label)
    return label

def oversample_data(data, y_combined):

    ros = RandomOverSampler(random_state=0)
    original_shape = data.shape
    train_data_reshaped = data.reshape(original_shape[0], -1)
    datas_resampled, y_resampled = ros.fit_resample(train_data_reshaped, y_combined)

    datas_resampled = datas_resampled.reshape((-1,) + original_shape[1:])
    labels_resampled = [float(label.split('_')[0]) for label in y_resampled]
    subject_labels_resampled = [float(label.split('_')[1]) for label in y_resampled]

    dataset = [
        [
            torch.tensor(datas_resampled[i], dtype=torch.float32),
            torch.tensor(labels_resampled[i], dtype=torch.int64),
            torch.tensor(subject_labels_resampled[i], dtype=torch.int64),
        ]
        for i in range(len(datas_resampled))
    ]

    return dataset


def load_data(data_path, label_type='A', subject_num=32):  # 读取训练数据
    datas = []
    labels = []
    subject_labels = []
    cnt = 0
    for home, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(".dat"):
                path = os.path.join(home, filename)
                print("loading : ", path)
                with open(path, 'rb') as file:
                    subject = pickle.load(file, encoding='latin1')
                label = subject['labels']
                label = label_selection(label, label_type=label_type)
                for i in range(len(label)):
                    base_signal = subject['data'][i,:,0:3*128]
                    for j in range(20):
                        temp = subject['data'][i, :, 3*128+j*3*128:3*128+j*3*128+3*128] - base_signal
                        datas.append(temp)
                        labels.append(label[i])
                        subject_labels.append(cnt)
                cnt = cnt+1
                if cnt == subject_num:
                    break

    return cnt, datas, labels, subject_labels

def get_loader(config):
    base_path = r"../../../DEAP_data_preprocessed_python/"

    # EEG:0
    # EOG:1
    # EMG:2

    cnt, datas, labels,subject_labels = load_data(base_path, label_type=config.label_type, subject_num=config.subject_num)
    datas = torch.tensor(np.array(datas), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.int64)
    subject_labels = torch.tensor(np.array(subject_labels), dtype=torch.int64)
    
    print(f'data shape: ', datas.shape)
    print(f'label shape: ', labels.shape)
    print(f'subject label shape: ', subject_labels.shape)
    
    train_datas = datas[0:800*(cnt-2)]
    train_labels = labels[0:800*(cnt-2)]
    train_subject_labels = subject_labels[0:800*(cnt-2)]
    train_combined_labels = [f"{label}_{subject_label}" for label, subject_label in zip(train_labels, train_subject_labels)]

    print(config.mode)
    config.data_len = len(datas)
    print(f'data len: ', config.data_len)

    train_dataset, dev_dataset, test_dataset = [], [], []
    train_dataset = oversample_data(train_datas, train_combined_labels)

    num_samples, num_nodes, _ = datas.shape

    dev_dataset = [[datas[s], labels[s], subject_labels[s]] for s in range(800 * (cnt-2), 800 * (cnt-1))]
    test_dataset = [[datas[s], labels[s], subject_labels[s]] for s in range(800 * (cnt-1), 800 * cnt)]

    print(f'train data len: ', len(train_dataset))
    print(f'dev data len', len(dev_dataset))
    print(f'test data len: ', len(test_dataset))

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''

        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
        subject_label = torch.tensor([sample[2] for sample in batch],dtype=torch.int64)

        eeg = pad_sequence([sample[0][0:32, :] for sample in batch])
        eog = pad_sequence([sample[0][32:34, :] for sample in batch])
        emg = pad_sequence([sample[0][34:36, :] for sample in batch])

        eeg = eeg.permute(1, 0, 2)
        eog = eog.permute(1, 0, 2)
        emg = emg.permute(1, 0, 2)


        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return eeg, eog, emg, labels, lengths, subject_label

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    return train_data_loader, dev_data_loader, test_data_loader