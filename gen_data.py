# -*- coding:utf-8 -*-
"""
File Name: tmp
Version:
Description:
Author: liuxuewen
Date: 2017/10/20 14:32
"""
import random

import tushare as ts
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def data_stock():
    end ='2017-10-20'
    start ='2000-01-01'
    instrument = '002230'
    #stock = ts.get_hist_data('instrument', start, end)
    data = ts.get_k_data(instrument, start, end)
    del data['code']
    data["amount"] = data["close"] * data["volume"]
    data['return'] = data['close'].shift(-5) / data['open'].shift(-1) - 1  # 计算未来5日收益率（未来第五日的收盘价/明日的开盘价）
    data.dropna(inplace=True)
    data['return'] = data['return'].apply(lambda x: np.where(x >= 0.2, 0.2, np.where(x > -0.2, x, -0.2)))  # 去极值
    #print(data)
    data.to_csv('data/data_stock.csv',encoding='utf-8')


def data_sinx():
    a = np.linspace(0, 100, 1000)
    b=np.sin(a)
    plt.figure()
    plt.plot(a, b, color='r')
    plt.show()
    df = pd.DataFrame({'x': a,
                       'y': b})
    # print(df)
    df.to_csv('data/data_sinx.csv')


def data_xsinx():
    a=np.linspace(0,100,1000)
    b=[x*np.sin(x) for x in a]
    plt.figure()
    plt.plot(a, b, color='r')
    plt.show()
    df=pd.DataFrame({'x':a,
                     'y':b})
    #print(df)
    df.to_csv('data/data_xsinx.csv')


if __name__ == '__main__':
    data_sinx()
    data_xsinx()
    data_stock()

