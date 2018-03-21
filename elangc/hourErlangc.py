# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:42:47 2018

@author: lfzhou
"""


'''
Possion 公式
http://www.excelfunctions.net/Excel-Poisson-Function.html
'''

import numpy as np
import pandas as pd
import math
from scipy.stats import poisson


def pmf(x, lambd):
    y = poisson.pmf(lambd, x)
    return y


def cdf(x, lamdb):
    y = poisson.cdf(lamdb, x)
    return y


def calSL(data, expectAanswerTime, slValue):
    i_list = data['avgtalktime'] * data['1小时呼入量'] / 3600  # talkForce  话务强度
    f_list = data['avgtalktime']  # avgtalktime
    n = expectAanswerTime
    result = []

    def probablity(g, i, j):
        denominator = pmf(i, g) + (1 - j) * cdf(i, g - 1)
        y = pmf(i, g) * 1.0 / denominator
        return y

    def avgWaitTime(g, j, k, f):
        denominator = g * (1 - j)
        y = k * f / denominator
        return y

    def serviceLevel(g, i, n, f, k):  # n为期望应答时间 指定为20s、
        expnum = -(g - i) * n / (f + 60)
        # y = list(map(lambda x,y: 1-y* math.exp(x),expnum,k))
        y = 1 - k * math.exp(expnum)
        return y

    for pos in range(len(data)):
        i = i_list[pos]
        f = f_list[pos]
        g = 0
        sl = 0
        while sl <= slValue:
            g += 1
            j = i / g  # eidrate 坐席占有率
            pro = probablity(g, i, j)
            sl = serviceLevel(g, i, n, f, pro) * 100

        awt = avgWaitTime(g, j, pro, f)
        res = [g, sl, pro, awt]
        result.append(res)

    result = pd.DataFrame(result)
    result['hour'] = data['hour']
    result.columns = ['hour', '人头', 'SL', '呼叫等待概率', '平均等待时长']
    return result


# if __name__ == "__main__":







