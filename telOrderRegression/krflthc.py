# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:15:34 2018

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
from sklearn.cross_validation import train_test_split


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


def dataMerge(language, ifdropna, SL_hour, Staff_hour, queue_hour):
    '''
    目的是合并多个sql运行出来的数据
    language : 输入需要的语言
    ifdropna ：是否需要删除缺失值
    sl_hour ： sl数据
    staff： 员工数目
    queue ： 等待时间
    '''
    SL_hour = lang2Tans(SL_hour, 'lang')
    dataSample = pd.merge(SL_hour, Staff_hour, how='inner', on=['lang', 'product', 'date'])
    # dataSample.drop(['date_y'],axis=1,inplace =True)
    # dataSample.rename(columns={'date_x':'date'},inplace=True)

    dataSample = pd.merge(dataSample, queue_hour, how='inner', on=['lang', 'product', 'date'])
    # dataset = pd.merge(dataSample,deal_hour,how='left',on = ['lang','date'])
    dataset = pulsWeekday(dataSample, 'date')

    langset = dataset[dataset.lang == language]
    # langpro = langset[langset['product']== product]
    # langpro.drop(['lang','date','hour','product'],axis=1,inplace=True)
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


def Modelconfig(data, Y):
    '''
   输出实际的模型的各项信息
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, random_state=1)
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


path = "D:\\Users\\lfzhou\\Desktop\\py\\SL_DAILY.xlsx"

SL_hour = read_excel(path, sheetname='SL_daily')
Staff_hour = read_excel(path, sheetname='Staff_daily')
queue_hour = read_excel(path, sheetname='queue_daily')
# deal_hour = read_excel(path,sheetname='deal_hour')

kr = dataMerge('KR', False, SL_hour, Staff_hour, queue_hour)

eidtotal = Staff_hour[(Staff_hour['product'] == 'total') & (Staff_hour['lang'] == 'KR')]
kr1 = pd.merge(kr, eidtotal, how='inner', on=['date', 'lang'])
kr1.drop(['product_y'], axis=1, inplace=True)

kr1.rename(columns={'eidno_x': 'eidPro', 'eidno_y': 'eidno', 'product_x': 'product'}, inplace=True)

kr = kr1
kr1.columns

kr.shape
kr['product'].value_counts()

data = kr[kr['product'] == '机票']

fea_cols = ['ServiceLevel', 'avgtalktime', 'eidno', 'avgQueue', 'weekday', 'connectedrate']

print('------- the origin analysis-----------')
sns.pairplot(data, x_vars=fea_cols, y_vars='totalcallincount', size=7, aspect=0.4, kind='reg')

# -----Y

Y = data.ix[:, 'totalcallincount'].values
# Y =list(map(lambda x: x*100,Y))

X = data[fea_cols]
X = X.fillna(X.mean())
# X = X.dropna(axis=0,how='any')
# np.where(np.isnan(data['avgdeal']))
X = ModelSet(X, False, -1)  # True代表需要归一化
# X = ModelSet(X,False,-1)


print('------- the model-----------')
X_train, X_test, Y_train, Y_test, y_pred, model = Modelconfig(X, Y)

print('-----the predict and test figure -------')
print('---len y_pred------', len(Y_test))
figureRoc(y_pred, Y_test, 40)
figureRoc(y_pred, Y_test, 40)

print('-----the Formula -------')
print(printFormula(fea_cols, model))

value = [70.0, 195, 24, 171, 1, 0.9]
print('ServiceLevel:', value[0], 'avgtalktime:', value[1], 'eidno:', value[2],
      'avgQueue:', value[3], 'weekday', value[4], 'connectedrate:', value[5])

CallCapicty = caluResut(fea_cols, model, 'value', value)
print('电话量:', CallCapicty)

ordVolume = int(CallCapicty / 0.35)
print('订单量:', ordVolume)

value = [70.0, 195, 28, 171, 1, 0.9]
print('ServiceLevel:', value[0], 'avgtalktime:', value[1], 'eidno:', value[2],
      'avgQueue:', value[3], 'weekday', value[4], 'connectedrate:', value[5])

CallCapicty = caluResut(fea_cols, model, 'value', value)
print('电话量:', CallCapicty)

ordVolume = int(CallCapicty / 0.35)
print('订单量:', ordVolume)

value = [70.0, 195, 39, 171, 1, 0.9]
print('ServiceLevel:', value[0], 'avgtalktime:', value[1], 'eidno:', value[2],
      'avgQueue:', value[3], 'weekday', value[4], 'connectedrate:', value[5])

CallCapicty = caluResut(fea_cols, model, 'value', value)
print('电话量:', CallCapicty)

ordVolume = int(CallCapicty / 0.35)
print('订单量:', ordVolume)

eidResume = [29, 36, 36, 45]
eidValue = list(map(lambda x: int(x * 0.7), eidResume))


def percentile(self, percentile):
    if len(self.sequence) < 1:
        value = None
    elif (percentile >= 100):
        sys.stderr.write('ERROR: percentile must be < 100.  you supplied: %s\n' % percentile)
        value = None
    else:
        element_idx = int(len(self.sequence) * (percentile / 100.0))
        self.sequence.sort()
        value = self.sequence[element_idx]
    return value