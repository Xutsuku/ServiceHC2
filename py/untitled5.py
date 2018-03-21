

from pandas import read_excel
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

def pulsWeekday(df,colname):    
    weekday = map(lambda x: x.isoweekday(),df[colname])
    df['weekday'] = list(weekday)
    return df

def lang2Tans(df,colname):    
    df[colname] = df[colname].replace({'英语':'EN','粤语':'HK','日语':'JP','韩语':'KR'})
    return df

def dropcols(df,colname):    
    df[colname] = df.drop(colname,axis=1,inplace=True)
    return df


#3 Roc曲线    #因为是预测类型的，所以没有所谓的那种东西出现 
def figureRoc(y_pred,Y_test,length):
    plt.figure()  
    plt.plot(range(length),y_pred[:length],'b',label="predict")  
    plt.plot(range(length),Y_test[:length],'r',label="test")  
    plt.legend(loc="upper right") #显示图中的标签  
    plt.xlabel("the number of connecttdRatte")
    plt.ylabel('value of connecttdRatte')
    plt.show()

def residual(y_pred,Y_test): 
    sum_mean=[]  
    for i in range(len(y_pred)):  
        sum_mean.append(abs((y_pred[i]-Y_test[i])))
    return sum_mean

def Rsquare(Y_test,y_pred):
    sum_mean=0  
    sum_residual = 0
    y_mean = np.sum(Y_test)/len(Y_test)
    for i in range(len(y_pred)):  
        sum_residual +=(y_pred[i]-Y_test[i])**2  
        sum_mean +=(y_pred[i]-y_mean)**2  
    Rs2 = 1 - sum_residual/sum_mean
    return  Rs2

def RMSE(Y_test,y_pred):
    sum_mean=0  
    for i in range(len(y_pred)):  
        sum_mean+=(y_pred[i]-Y_test[i])**2  
    rmse=np.sqrt(sum_mean/len(y_pred))  
    # calculate RMSE by hand  
    return rmse

def printFormula(fea_cols,model):
    result = zip(fea_cols, model.coef_)    
    coef2 =dict()
    for i in range(len(fea_cols)):
        z= result.__next__()
        coef2[z[0]] = z[1]
    formula = 'connectedRate ='    
    for i in coef2.keys():
        if coef2[i] >= 0:
            coef = str(coef2[i])
            formula += '+'+ coef + '*' + '【'+ i+'】'            
        else:
            coef = str(coef2[i])
            formula += coef + '*' + '【'+ i+'】'      
    return formula
               
def caluResut(fea_cols,model,colname,value):
    result = zip(fea_cols, model.coef_)    
    coef2 =dict()
    for i in range(len(fea_cols)):
        z= result.__next__()
        coef2[z[0]] = z[1]
    #df = pd.Series(coef2)
    df =pd.DataFrame(list(coef2.items()), columns=['key','coef'])
    df[colname] = value    
    result = sum(df.coef*df.value) 
    return result

def dataMerge(language,ifdropna,SL_hour,Staff_hour,queue_hour):
    SL_hour = lang2Tans(SL_hour,'lang')   
    dataSample = pd.merge(SL_hour, Staff_hour,how='inner',on=['lang','product','date'])
    #dataSample.drop(['date_y'],axis=1,inplace =True)
    #dataSample.rename(columns={'date_x':'date'},inplace=True)
    
    dataSample = pd.merge(dataSample, queue_hour,how='inner',on =['lang','product','date'])
    #dataset = pd.merge(dataSample,deal_hour,how='left',on = ['lang','date'])
    dataset = pulsWeekday(dataSample,'date')  

    langset = dataset[dataset.lang == language]
    #langpro = langset[langset['product']== product] 
    #langpro.drop(['lang','date','hour','product'],axis=1,inplace=True)
    if ifdropna:
        langset = langset.dropna(axis=0)
    else:
        pass
    return langset

def ModelSet(data,ifNorm,colindex):
    datavalues = data[fea_cols].values
    if ifNorm and  colindex is not None:
        #归一化
        data = data.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        # 对weekday 做标签
        encoder = LabelEncoder()
        data[:,colindex] = encoder.fit_transform(datavalues[:,colindex])
    else:
        pass
    return data


def Modelconfig(data,Y): 

    X_train,X_test,Y_train,Y_test = train_test_split(data,Y, random_state=1)
    print ('-----Shape of X_train,X_test,Y_train,Y_test-------')
    print( X_train.shape,len(Y_train),X_test.shape,len(Y_test))
    
    from sklearn.linear_model import LinearRegression  
    linreg = LinearRegression()  
    model=linreg.fit(X_train, Y_train)  
    print ('------- the model-----------') 
    print(model)
    print(linreg.intercept_)  
    print(linreg.coef_) 
    
    y_pred = linreg.predict(X_test)  
    print (y_pred)
    
    print ("---Rsquare is:" ,Rsquare(Y_test,y_pred))
    print ("---RMSE is:" ,RMSE(Y_test,y_pred))
    res = residual(y_pred,Y_test)
    print ("---top 20 residual is:", res[:20])
    
    return  X_train,X_test,Y_train,Y_test,y_pred,model
        


path ="D:\\Users\\lfzhou\\Desktop\\SL_DAILY.xlsx"

SL_hour = read_excel(path,sheetname='SL_daily')
Staff_hour = read_excel(path,sheetname='Staff_daily')
queue_hour = read_excel(path,sheetname='queue_daily')
#deal_hour = read_excel(path,sheetname='deal_hour') 

kr = dataMerge('KR',False,SL_hour,Staff_hour,queue_hour)
en.columns
en.shape
en['product'].value_counts()


data = kr[kr['product'] =='机票']



fea_cols = ['totalcallincount','ConnectedInTwenty', 'abandonedintwenty', 'avgtalktime',
       'eidno', 'avgQueue']

print ('------- the origin analysis-----------')    
sns.pairplot(data,x_vars=fea_cols,y_vars ='connectedrate',size=7,aspect=0.4,kind='reg')
 
#-----Y

Y = data.ix[:,'connectedrate'].values
Y =list(map(lambda x: x*100,Y))

#-----x
X = data[fea_cols]
X = X.fillna(X.mean())
#X = X.dropna(axis=0,how='any')
#np.where(np.isnan(data['avgdeal']))
X = ModelSet(X,True,-1)   # True代表需要归一化 
#X = ModelSet(X,False,-1) 
    

print ('------- the model-----------') 
X_train,X_test,Y_train,Y_test,y_pred,model = Modelconfig(X,Y) 
 
  
print ('-----the predict and test figure -------') 
print('---len y_pred------', len(Y_test))
figureRoc(y_pred,Y_test,40)
figureRoc(y_pred,Y_test,100)
   
print ('-----the Formula -------')
print(printFormula(fea_cols,model))  
value = range(7)
caluResut(fea_cols,model,'value',value) 

