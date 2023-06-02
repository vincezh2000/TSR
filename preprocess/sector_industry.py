import argparse
from datetime import datetime
import copy
import json
import numpy as np
import operator
import os
import pandas as pd

class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname):
        # 提取股票代码
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(selected_tickers))
        # 转换成索引
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        # 提取行业数据
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))
        # 提取有效行业类别数据，如果没有对应的股票行业则将其去掉
        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys():
            if len(industry_tickers[industry]) > 1:
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
        # 初始化单位矩阵，后续将作为转换成热编码用（+1，如果没有归属的股票则属于else）
        # 创建一个独热编码矩阵。每行只有一个元素为1，其他元素为0，一个n x n 的对角矩阵
        one_hot_industry_embedding = np.identity(valid_industry_count + 1,
                                                 dtype=int)
        # 股票数*股票数*行业数，存储关联关系矩阵，并不是邻接矩阵
        # 每个元素[i][j][k]表示第i个股票与第j个股票之间的关联关系，（并且其行业类别为k，其对应的值为1）
        ticker_relation_embedding = np.zeros(
            [len(selected_tickers), len(selected_tickers),
             valid_industry_count + 1], dtype=int)
        print(ticker_relation_embedding[0][0].shape)
        # 循环将json文件里的数据存储到三维矩阵ticker_relation_embedding中
        # 首先遍历有效行业
        for industry in valid_industry_index.keys():
            # 提取行业内股票名称
            cur_ind_tickers = industry_tickers[industry]
            # 行业内股票数小于一就跳过
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry) # What the fk lol
                continue

            # 提取行业编号。 e.g edp -> 0
            ind_ind = valid_industry_index[industry]
            # 循环遍历该行业下的股票
            for i in range(len(cur_ind_tickers)):
                # 提取出股票索引。cur_ind_tickers[i]是行业内股票名。
                # ticker_index是股票名对应的全局股票索引值
                left_tic_ind = ticker_index[cur_ind_tickers[i]]
                # 提取出股票编号
                ticker_relation_embedding[left_tic_ind][left_tic_ind] = \
                    copy.copy(one_hot_industry_embedding[ind_ind])
                # 为什么还要设为1？
                ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1
                for j in range(i + 1, len(cur_ind_tickers)):
                    right_tic_ind = ticker_index[cur_ind_tickers[j]]
                    ticker_relation_embedding[left_tic_ind][right_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    ticker_relation_embedding[right_tic_ind][left_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    # print(right_tic_ind)

        # handle shit industry and n/a tickers
        # 如果有股票由于只属于shit industry, 最后一行没有赋予1，就执行一下代码检查一遍并赋予1
        # 综上，如果embedding[i][i]不是shit，那么其所属行业列和最后一列都为1
        # 如果为shit，那么只有最后一列为1
        for i in range(len(selected_tickers)):
            ticker_relation_embedding[i][i][-1] = 1
        print(ticker_relation_embedding.shape)
        np.save(self.market_name + '_industry_relation',
                ticker_relation_embedding)


if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '../data/google_finance'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        os.path.join('../data/relation/sector_industry/',
                     processor.market_name + '_industry_ticker.json'),
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    )