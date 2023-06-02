import copy
import numpy as np
import os


def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    # 循环遍历股票列表，提取相关数据
    for index, ticker in enumerate(tickers):
        # 提取股票历史价格（即预处理后的数据）
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            # 若果索引为0，就初始化数组，全为0【股票个数，时间跨度/天，特征数（第一维不算）】
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            # 全为1，若为0，则代表缺失 【股票个数，T】
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            # 【股票个数，T】
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            # 【股票个数，T】目的是计算预测的收益率
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            # 如果收盘价缺失，mask记为0
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            # steps=1(default)
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                # return =（今日-昨日）/昨日
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            # 如果数据缺失填充为1.1  （可能防止什么？避免出现离群点）
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        # 赋值特征列
        eod_data[index, :, :] = single_EOD[:, 1:]
        # 赋值收盘价close
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # np.sum用于统计每个位置的非零元素个数。
    # np.equal()函数对两个矩阵逐元素进行比较。元素为True表示对应位置的关联数据是空的，而False表示对应位置的关联数据是非空的
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    # np.where(condition, x, y)  condition 中的元素为 True 时，选择x
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # handle missing data
                if row < 3:
                    # eod_data[index, row] = 0.0
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            # print(index, row, i, eod_data[index][row])
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
                    # print(index, row, eod_data[index][row])
            else:
                eod_data[index][row] = single_EOD[row][-1]
        # print('test point')
    np.save(market_name + '_sfm_data', eod_data)