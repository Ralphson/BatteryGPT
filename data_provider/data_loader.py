import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    # padding mask
    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]], dtype=object)  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

        self.mins = 9999
        for i in self.timeseries:
            if i.shape[0] < self.mins:
                self.mins = i.shape[0]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask



class Dataset_Battery(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, cutting_rate=1.2, drop_bid=False, seq_limit=0):
        """
        ~~ 不能在轮次维度划分,电池维度有寿命衰减存在 ~~

        - 可以,直接上shuffle,对电池和轮次,因为有bid和轮次id可以标识电池充电次数信息
        - 对于一个轮次的短时序列:
            1. 拆成`label_len`和`pred_len`,对应总长度`seq_len`会覆盖住所有电池中
            最长的充电序列,`seq_len-pred_len-label_len`的部分用掩码遮住
        """
        if size == None:
            self.seq_len = 18    # 训练长度
            self.label_len = 9  # 重叠部分,用于预测序列回看知识
            self.pred_len = 30   # 预测
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.drop_bid = drop_bid    # TODO:battery_id可能需要独热编码,考虑drop或者不进行归一化
        self.cutting_rate = cutting_rate
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seq_limit = seq_limit

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 按照电池+轮次打乱
        indexed_df = df_raw.set_index(['battery_id', 'Cycle_Index'])
        indexed_df['luncishu'] = df_raw.groupby(['battery_id', 'Cycle_Index']).count()['cycle_test_time']
        indexed_df = indexed_df[indexed_df['luncishu'] > self.seq_limit]   # 只保留48以上的轮次
        indexed_df = indexed_df.drop('luncishu', axis=1)        # 删除luncishu
        shuffled_index = indexed_df.index.unique().values
        random.shuffle(shuffled_index)
        shuffled_index = pd.Index(shuffled_index)

        # 找到对应数据集的位置
        border1s = [0, 0.6, 0.7]
        border2s = [0.6, 0.7, 1]
        def get_data(df, shuffled_index, a, b):
            border1 = int(a*shuffled_index.shape[0])
            border2 = int(b*shuffled_index.shape[0])
            flag = shuffled_index[border1:border2]
            df = df.loc[flag]
            return df

        # 获取对应模式的数据
        df = get_data(indexed_df, shuffled_index, border1s[self.set_type], border2s[self.set_type])

        # 记录恢复标识
        recover_index = df.index
        self.recover_index_list = []    # 数字index
        self.recover_name_list = []     # 原始multiindex标志
        a = pd.DataFrame(range(recover_index.shape[0]), index=recover_index)
        groups = a.groupby(level=[0,1], sort=False)
        for name, group in groups:
            self.recover_index_list.append(group[0].to_numpy())
            self.recover_name_list.append(name)
        self.recover_index_list = np.array(self.recover_index_list, dtype=object)
        self.recover_name_list = np.array(self.recover_name_list, dtype=object)

        # unstack index
        df = df.reset_index()   

        # 删除bid
        if self.drop_bid:
            df = df.drop('battery_id', axis=1)

        # 归一化
        scaler = StandardScaler()
        if self.scale:
            train_data = get_data(indexed_df, shuffled_index, border1s[0], border2s[0])
            train_data = train_data.reset_index()
            if self.drop_bid:
                train_data = train_data.drop('battery_id', axis=1)
            scaler.fit(train_data.values)
            self.data = scaler.transform(df.values)  
        else:
            self.data = df.values

        # 不等长的数据
        self.data = np.array([self.data[index] for index in self.recover_index_list], dtype=object)
        print('data-{} load completed: {} lunci'.format(self.set_type, self.data.shape))

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 11))
        insample_mask = np.zeros((self.seq_len, 11))
        outsample = np.zeros((self.label_len + self.pred_len, 1))
        outsample_mask = np.zeros((self.label_len + self.pred_len, 1))

        seq = self.data[index]
        try:
            low = self.seq_len
            high = len(seq) - self.pred_len
            cut_point = np.random.randint(low=low,
                                          high=high,
                                          size=1)[0]    # 切分点,
        except:
            raise ValueError("index={}, low={}, high={}".format(index, low, high), len(seq))

        insample = seq[cut_point - self.seq_len: cut_point]
        outsample = seq[cut_point - self.label_len: cut_point + self.pred_len, -1]
        outsample = outsample[:, np.newaxis]

        return insample, outsample, insample_mask, outsample_mask


    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Masked_Battery(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, cutting_rate=1.2, drop_bid=False, seq_limit=0):
        if size == None:
            self.seq_len = 24    # 训练长度
            self.label_len = 12  # 重叠部分,用于预测序列回看知识
            self.pred_len = 36   # 预测
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.drop_bid = drop_bid    # TODO:battery_id可能需要独热编码,考虑drop或者不进行归一化
        self.cutting_rate = cutting_rate
        self.seq_limit = seq_limit
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 按照电池+轮次打乱
        indexed_df = df_raw.set_index(['battery_id', 'Cycle_Index'])
        indexed_df['luncishu'] = df_raw.groupby(['battery_id', 'Cycle_Index']).count()['cycle_test_time']
        indexed_df = indexed_df[indexed_df['luncishu'] > self.seq_limit]   # 只保留<48>以上的轮次
        indexed_df = indexed_df.drop('luncishu', axis=1)        # 删除luncishu
        shuffled_index = indexed_df.index.unique().values
        random.shuffle(shuffled_index)
        shuffled_index = pd.Index(shuffled_index)

        # 找到对应数据集的位置
        border1s = [0, 0.6, 0.7]
        border2s = [0.6, 0.7, 1]
        def get_data(df, shuffled_index, a, b):
            border1 = int(a*shuffled_index.shape[0])
            border2 = int(b*shuffled_index.shape[0])
            flag = shuffled_index[border1:border2]
            df = df.loc[flag]
            return df

        # 获取对应模式的数据
        df = get_data(indexed_df, shuffled_index, border1s[self.set_type], border2s[self.set_type])

        # 记录恢复标识
        recover_index = df.index
        self.recover_index_list = []    # 数字index
        self.recover_name_list = []     # 原始multiindex标志
        a = pd.DataFrame(range(recover_index.shape[0]), index=recover_index)
        groups = a.groupby(level=[0,1], sort=False)
        for name, group in groups:
            self.recover_index_list.append(group[0].to_numpy())
            self.recover_name_list.append(name)
        self.recover_index_list = np.array(self.recover_index_list, dtype=object)
        self.recover_name_list = np.array(self.recover_name_list, dtype=object)

        # unstack index
        df = df.reset_index()   

        # 删除bid
        if self.drop_bid:
            df = df.drop('battery_id', axis=1)

        # 归一化
        scaler = StandardScaler()
        if self.scale:
            train_data = get_data(indexed_df, shuffled_index, border1s[0], border2s[0])
            train_data = train_data.reset_index()
            if self.drop_bid:
                train_data = train_data.drop('battery_id', axis=1)
            scaler.fit(train_data.values)
            self.data = scaler.transform(df.values)  
        else:
            self.data = df.values

        # 等长数据
        self.data = np.array([self.data[index] for index in self.recover_index_list], dtype=object)
        print('data-{} load completed: {}'.format(self.set_type, self.data.shape))

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 11))
        insample_mask = np.zeros((self.seq_len, 11))
        outsample = np.zeros((self.label_len + self.pred_len, 1))
        outsample_mask = np.zeros((self.label_len + self.pred_len, 1))

        seq = self.data[index]
        try:
            low = max(1, len(seq) - int(self.pred_len * self.cutting_rate))
            high = len(seq)
            cut_point = np.random.randint(low=low,
                                          high=high,
                                          size=1)[0]    # 切分点,
        except:
            raise ValueError("index={}, low={}, high={}".format(index, low, high), len(seq))

        # 输入padding mask
        insample_window = seq[max(0, cut_point - self.seq_len): cut_point]
        insample[-len(insample_window):, :] = insample_window
        insample_mask[-len(insample_window):, :] = 1.0

        # 输出padding mask
        outsample_window = seq[max(1, cut_point-self.label_len): min(len(seq), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window[:, -1]
        outsample_mask[:len(outsample_window), 0] = 1.0

        return insample, outsample, insample_mask, outsample_mask


    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_Masked_Battery_from_0(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv',
                 target='OT', scale=1, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, cutting_rate=1.2, drop_bid=False, seq_limit=0):
        if size == None:
            self.seq_len = 24    # 训练长度
            self.label_len = 12  # 重叠部分,用于预测序列回看知识
            self.pred_len = 48   # 预测
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.drop_bid = drop_bid    # TODO:battery_id可能需要独热编码,考虑drop或者不进行归一化
        self.cutting_rate = cutting_rate
        self.seq_limit = seq_limit
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 按照电池+轮次打乱
        indexed_df = df_raw.set_index(['battery_id', 'Cycle_Index'])
        indexed_df['luncishu'] = df_raw.groupby(['battery_id', 'Cycle_Index']).count()['cycle_test_time']
        indexed_df = indexed_df[indexed_df['luncishu'] > self.seq_limit]   # 只保留<48>以上的轮次
        indexed_df = indexed_df.drop('luncishu', axis=1)        # 删除luncishu
        shuffled_index = indexed_df.index.unique().values
        random.shuffle(shuffled_index)
        shuffled_index = pd.Index(shuffled_index)

        # 找到对应数据集的位置
        border1s = [0, 0.6, 0.7]
        border2s = [0.6, 0.7, 1]
        def get_data(df, shuffled_index, a, b):
            border1 = int(a*shuffled_index.shape[0])
            border2 = int(b*shuffled_index.shape[0])
            flag = shuffled_index[border1:border2]
            df = df.loc[flag]
            return df

        # 获取对应模式的数据
        df = get_data(indexed_df, shuffled_index, border1s[self.set_type], border2s[self.set_type])

        # 记录恢复标识
        recover_index = df.index
        self.recover_index_list = []    # 数字index
        self.recover_name_list = []     # 原始multiindex标志
        a = pd.DataFrame(range(recover_index.shape[0]), index=recover_index)
        groups = a.groupby(level=[0,1], sort=False)
        for name, group in groups:
            self.recover_index_list.append(group[0].to_numpy())
            self.recover_name_list.append(name)
        self.recover_index_list = np.array(self.recover_index_list, dtype=object)
        self.recover_name_list = np.array(self.recover_name_list, dtype=object)

        # unstack index
        df = df.reset_index()   

        # 删除bid
        if self.drop_bid:
            df = df.drop('battery_id', axis=1)

        # 归一化
        scaler = StandardScaler()
        if self.scale == 1:
            print('do scaling...')
            train_data = get_data(indexed_df, shuffled_index, border1s[0], border2s[0])
            train_data = train_data.reset_index()
            if self.drop_bid:
                train_data = train_data.drop('battery_id', axis=1)
            scaler.fit(train_data.values)
            self.data = scaler.transform(df.values)  
        else:
            self.data = df.values

        # 等长数据
        self.data = np.array([self.data[index] for index in self.recover_index_list], dtype=object)
        print('data-{} load completed: {} from 0'.format(self.set_type, self.data.shape))

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 11))
        insample_mask = np.zeros((self.seq_len, 11))
        outsample = np.zeros((self.label_len + self.pred_len, 1))
        outsample_mask = np.zeros((self.label_len + self.pred_len, 1))

        seq = self.data[index]

        # 输入padding mask
        insample = seq[0: self.seq_len]

        # 输出padding mask
        outsample_window = seq[self.seq_len-self.label_len: min(len(seq), self.seq_len + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window[:, -1]
        outsample_mask[:len(outsample_window), 0] = 1.0

        return insample, outsample, insample_mask, outsample_mask


    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





if __name__=="__main__":
    mydataset = Dataset_Masked_Battery(
        root_path='./dataset/my/',
        data_path='trimmed_LX3_ss0_se100_cr05_C_V_T_vs_CE.csv',
    )
