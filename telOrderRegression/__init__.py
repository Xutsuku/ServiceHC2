
# -*- coding: utf-8 -*-
# ----------------Main------------------


from telOrderRegression.orderTelVolume import telorderVolume,datasetMerge
import pandas as pd
from pandas import read_excel

lang ='KR'
#path = "D:\\Users\\lfzhou\\Desktop\\py\\SL_DAILY.xlsx"
# datasetMerge(path,lang)
path = "D:\\Users\\lfzhou\\Desktop\\py\\SL_DAILY.xlsx"
path = 'd:\\Users\\lfzhou\\Desktop\\服务运营HC 流程化\\ErlangC.xlsx'
lang = 'KR'
data = datasetMerge(path, lang, sheet=['SL_Daily', 'eidNo_Daily', 'Deal_Daily'])

fea_cols = ['ServiceLevel', 'avgtalktime', 'eidno', 'weekday', 'connectedrate']

value = [70.0, 195, 24, 1, 0.9]
telordRate = 0.35

telorderVolume(data, fea_cols, value, telordRate)

