# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:36:32 2018

@author: lfzhou
"""

from pandas import read_excel
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



def pulsWeekday(df, colname):
    weekday = map(lambda x: x.isoweekday(), df[colname])
    df['weekday'] = list(map(lambda x: IsweekdayInt(x), weekday))
    return df


def lang2Tans(df, colname):
    df[colname] = df[colname].replace({'英语': 'EN', '粤语': 'HK', '日语': 'JP', '韩语': 'KR'})
    return df


def dropcols(df, colname):
    df[colname] = df.drop(colname, axis=1, inplace=True)
    return df


# 3 Roc曲线    #因为是预测类型的，所以没有所谓的那种东西出现
def figureRoc(y_pred, Y_test, length):
    plt.figure()
    plt.plot(range(length), y_pred[:length], 'b', label="predict")
    plt.plot(range(length), Y_test[:length], 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of connecttdRatte")
    plt.ylabel('value of connecttdRatte')
    plt.show()


def residual(y_pred, Y_test):
    sum_mean = []
    for i in range(len(y_pred)):
        sum_mean.append(abs((y_pred[i] - Y_test[i])))
    return sum_mean


def Rsquare(Y_test, y_pred):
    sum_mean = 0
    sum_residual = 0
    y_mean = np.sum(Y_test) / len(Y_test)
    for i in range(len(y_pred)):
        sum_residual += (y_pred[i] - Y_test[i]) ** 2
        sum_mean += (y_pred[i] - y_mean) ** 2
    Rs2 = 1 - sum_residual / sum_mean
    return Rs2


def RMSE(Y_test, y_pred):
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - Y_test[i]) ** 2
    rmse = np.sqrt(sum_mean / len(y_pred))
    # calculate RMSE by hand
    return rmse


def printFormula(fea_cols, model):
    '''
    将当前的公式打印出来
    '''
    result = zip(fea_cols, model.coef_)
    coef2 = dict()
    for i in range(len(fea_cols)):
        z = result.__next__()
        coef2[z[0]] = z[1]
    formula = 'connectedRate ='
    for i in coef2.keys():
        if coef2[i] >= 0:
            coef = str(coef2[i])
            formula += '+' + coef + '*' + '【' + i + '】'
        else:
            coef = str(coef2[i])
            formula += coef + '*' + '【' + i + '】'
    return formula


def caluResut(fea_cols, model, colname, value):
    '''
    给定一组值算模型下面的预测的电话数据
    fea_cols:
    model
    colname : 算出来的值的列名 ‘value’
    value  给定的一组值
    '''
    result = zip(fea_cols, model.coef_)
    coef2 = dict()
    for i in range(len(fea_cols)):
        z = result.__next__()
        coef2[z[0]] = z[1]
    # df = pd.Series(coef2)
    df = pd.DataFrame(list(coef2.items()), columns=['key', 'coef'])
    df[colname] = value
    result = sum(df.coef * df.value)
    return result


def IsweekdayInt(x):
    ''' 打标签：工作日和周末，1为工作日，0 为周末'''
    if x >= 6:
        y = 0
    else:
        y = 1
    return y


def dataMerge(language, ifdropna, SL, Staff, Other, key=['lang', 'product', 'date']):
    '''
    目的是合并多个sql运行出来的数据
    language : 输入需要的语言
    ifdropna ：是否需要删除缺失值
    sl_hour ： sl数据
    staff： 员工数目
    Other： 等待时间等其他因素，可填可不填,默认可以是空df []
    key :如果是按照hour的话，需要变为key=['lang','product','date','hour']
    '''
    SL = lang2Tans(SL, 'lang')
    dataSample = pd.merge(SL, Staff, how='inner', on=key)

    if len(Other) > 0:
        dataSample = pd.merge(dataSample, Other, how='inner', on=key)
        dataset = pulsWeekday(dataSample, 'date')
    else:
        dataset = pulsWeekday(dataSample, 'date')
    langset = dataset[dataset.lang == language]

    if ifdropna:
        langset = langset.dropna(axis=0)
    else:
        pass
    return langset


def ModelSet(data, ifNorm, colindex):
    '''
    多元回归分析
    data : 数据 自变量因变量等在一起的dataframe
    ifNorm : 是否要对数据进行归一化
    colindex : 自变量选取的特征
    '''
    datavalues = data[fea_cols].values
    if ifNorm and colindex is not None:
        # 归一化
        data = data.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        # 对weekday 做标签
        encoder = LabelEncoder()
        data[:, colindex] = encoder.fit_transform(datavalues[:, colindex])
    else:
        pass
    return data


def Modelconfig(data, Y, size=0.8):
    '''
     输出实际的模型的各项信息 :
     coef
     predict
     rmse
     residual
    '''
    length = int(data.shape[0] * 0.8)
    # X_train,X_test,Y_train,Y_test = train_test_split(data,Y, random_state=1)
    X_train, X_test, Y_train, Y_test = data[0:length], data[length:-1], Y[0:length], Y[length:-1]
    print('-----Shape of X_train,X_test,Y_train,Y_test-------')
    print(X_train.shape, len(Y_train), X_test.shape, len(Y_test))

    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    model = linreg.fit(X_train, Y_train)
    print('------- the model-----------')
    print(model)
    print(linreg.intercept_)
    print(linreg.coef_)

    y_pred = linreg.predict(X_test)
    print(y_pred)

    print("---Rsquare is:", Rsquare(Y_test, y_pred))
    print("---RMSE is:", RMSE(Y_test, y_pred))
    res = residual(y_pred, Y_test)
    print("---top 20 residual is:", res[:20])

    return X_train, X_test, Y_train, Y_test, y_pred, model


def datasetMerge(path, lang, product='机票', ifdropna=False, sheet=['SL_daily', 'Staff_daily', 'queue_daily'],
                 key=['date', 'lang']):
    '''
    读取excel的数据并合并为最后的input data
    sheet3 为avgdeal 或者 queun 这两个变量，具体名字可以更改
    key 为sl和staff 的total join 时，数据为daily 还是hour 格式，如果是hour 则更改为['date','lang','hour']
    '''
    SL = read_excel(path, sheetname=sheet[0])
    Staff = read_excel(path, sheetname=sheet[1])
    Other = read_excel(path, sheetname=sheet[2])
    dataset = dataMerge(lang, ifdropna, SL, Staff, Other)

    eidtotal = Staff[(Staff['product'] == 'total') & (Staff['lang'] == lang)]
    dataset = pd.merge(dataset, eidtotal, how='inner', on=key)
    dataset.drop(['product_y'], axis=1, inplace=True)

    dataset.rename(columns={'eidno_x': 'eidPro', 'eidno_y': 'eidno', 'product_x': 'product'}, inplace=True)

    print('shape:\n', dataset.shape)
    print('product value counts:', dataset['product'].value_counts())

    data = dataset[dataset['product'] == product]
    print(data.head())
    return data


def telorderVolume(data, fea_cols, value, telordRate, ifdropna=False, y_var='totalcallincount', ifnorm=False):
    print('------- the origin analysis pairplot-----------')
    sns.pairplot(data, x_vars=fea_cols, y_vars=y_var, size=7, aspect=0.4, kind='reg')

    # -----Y
    Y = data.ix[:, 'totalcallincount'].values

    X = data[fea_cols]
    X = X.fillna(X.mean())

    X = ModelSet(X, ifnorm, -1)  # True代表需要归一化

    print('-------------- the model-----------')
    X_train, X_test, Y_train, Y_test, y_pred, model = Modelconfig(X, Y)

    print('---------------the predict and test figure -------')
    print('len y_pred:', len(Y_test))
    figureRoc(y_pred, Y_test, len(Y_test))

    print('------------the Formula -------')
    print(printFormula(fea_cols, model))

    print('---------------Value ------------------')
    for i, pos in enumerate(fea_cols):
        print(pos, value[i])

    CallCapicty = caluResut(fea_cols, model, 'value', value)
    print('电话量:', CallCapicty)

    ordVolume = int(CallCapicty / telordRate)
    print('订单量:', ordVolume)

    return printFormula(fea_cols, model), ordVolume, CallCapicty




