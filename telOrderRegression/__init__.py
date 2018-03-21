
# -*- coding: utf-8 -*-
# ----------------Main------------------


from telOrderRegression.orderTelVolume import lang2Tans,ModelSet,printFormula
import pandas as pd
from pandas import read_excel


path = "d:\\Users\\lfzhou\\Desktop\\服务运营HC 流程化\\ErlangC.xlsx"

def zlf_readExcel(path,lang,fea_cols,prodcut):

    SL = read_excel(path, sheetname='SL_Daily')
    Staff = read_excel(path, sheetname='eidNo_Daily')
    Deal = read_excel(path, sheetname='Deal_Daily')
    Queue =  read_excel(path, sheetname='Queue_Daily')

    SL = lang2Tans(SL, 'lang')
    dataSample = pd.merge(SL, Staff, how='inner', on=['lang', 'product', 'date'])


    kr = dataMerge('KR', False, SL, Staff, Deal)
    kr.columns
    kr.shape
    kr['product'].value_counts()

    data = kr[kr['product'] == '机票']

    fea_cols = ['ServiceLevel', 'avgtalktime', 'eidno', 'avgDeal', 'weekday', 'connectedrate']

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
figureRoc(y_pred, Y_test, 100)

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

