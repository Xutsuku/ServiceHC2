# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:55:23 2018

@author: lfzhou
"""


import datetime 
import time


一、date模块

字符串模式：Wed Feb 15 11:40:23 2017
时间戳格式：1463846400.0
结构化时间元祖：time.struct_time(tm_year=2017, tm_mon=2, tm_mday=15, tm_hour=11, tm_min=40, tm_sec=23, tm_wday=2, tm_yday=46, tm_isdst=0)

时间元祖是时间类型转化的关键


print(time.asctime()) #Wed Feb 15 11:15:21 2017
print(time.localtime()) 
#time.struct_time(tm_year=2018, tm_mon=2, tm_mday=7, 
#     tm_hour=17, tm_min=55, tm_sec=57, tm_wday=2, tm_yday=38, tm_isdst=0)

time.ctime() #'Wed Feb  7 17:56:27 2018'  这里是字符
time.time()  # 1517997407.641

#--------------时间类型转换-------------------------
time.gmtime()  #时间元祖
#Out[45]: time.struct_time(tm_year=2018, tm_mon=2, tm_mday=7, 
   #tm_hour=10, tm_min=3, tm_sec=9, tm_wday=2, tm_yday=38, tm_isdst=0)

string_2_struct = time.strptime("2016/05/22","%Y/%m/%d")  #将字符串转为时间元祖
print(time.gmtime(time.time()))  #将时间粗转元祖
#time.struct_time(tm_year=2018, tm_mon=2, tm_mday=7,
#     tm_hour=9, tm_min=59, tm_sec=17, tm_wday=2, tm_yday=38, tm_isdst=0)

struct_2_stamp = time.mktime(string_2_struct)  # 将元祖转时间粗 1463846400.0

print( time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))  #2018-02-07 10:03:57  
# 将元祖转为指定的字符串格式

#------------其他用途的时间函数-------------------------
print(time.process_time())#处理器计算时间
time.sleep(2) #睡眠指定的时间


'''
二、datetime模块的使用和说明
datetime模块用于是date和time模块的合集，datetime有两个常量，MAXYEAR和MINYEAR，分别是9999和1.

datetime模块定义了5个类，分别是

1.datetime.date：表示日期的类
2.datetime.datetime：表示日期时间的类
3.datetime.time：表示时间的类
4.datetime.timedelta：表示时间间隔，即两个时间点的间隔
5.datetime.tzinfo：时区的相关信息
'''
from datetime import datetime,date,timedelta,time
#---1.datetime.date：表示日期的类
today = date.today()

date.fromtimestamp(1457877369.650549) #datetime.date(2016, 3, 13)
date.fromordinal(1)  #将天数+最小日期 转换成日期输出    0001-01-01 

date.min
date.max
date.resolution


d = date(2017,7,1)  #定义好date 

d.replace(day=30) # 替换某个日期
d.weekday()  #.date.weekday()返回当前日期是所在周的第几天  0 表示周一 6 表示周日
d.isoweekday() #1 表示周一 7 表示周日

print(d.toordinal()); #该日期距离最小日期的天数  736330  
print(d.weekday());#返回当前日期是所在周的第几天  0 表示周一 6 表示周日  
print(d.isoweekday());#返回当前日期是所在周的第几天  1 表示周一 7 表示周日  
print(d.isocalendar());#返回格式如(year，month，day)的元组    
print(d.isocalendar()[1]);#返回该日期是这一年中的第几周  
print(d.isocalendar()[2]);#返回该日期是周几  
print(d.isoformat());#返回 ISO 8601格式  YYYY-MM-DD  
print(d.strftime("%d/%m/%y"));#04/01/17  
print(d.__format__("%d/%m/%y"));#04/01/17  
print(d.ctime());#Wed Jan  4 00:00:00 2017  




#---2.datetime.datetime：表示日期时间的类
a = datetime.datetime(2017, 4, 16)  # datetime.datetime(2017, 4, 16, 0, 0)
b = datetime.datetime(2017, 4, 16, 21, 21, 20, 871000)
b-a
(b-a).seconds
(b-a).days
(b-a).total_seconds()




                  




datetime是基于time模块封装的，使用起来更加友好，但是执行效率略低。
datetime里有四个重要的类：datetime、date、time、timedelta

#1、datetime类：

　　#创建datetime对象：

　　　　datetime.datetime (year, month, day[ , hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] ] )

　　　　datetime.fromtimestamp(cls, timestamp, tz=None)

　　　　datetime.utcfromtimestamp(cls, timestamp)

　　　　datetime.now(cls, tz=None)

　　　　datetime.utcnow(cls)

　　　　datetime.combine(cls, datetime.date, datetime.time)

　　#datetime的实例方法：

　　　　datetime.year、month、day、hour、minute、second、microsecond、tzinfo：
　　　　datetime.date()：获取date对象；
　　　　datetime.time()：获取time对象；
　　　　datetime. replace ([ year[ , month[ , day[ , hour[ , minute[ , second[ , microsecond[ , tzinfo] ] ] ] ] ] ] ])：
　　　　datetime. timetuple ()
　　　　datetime. utctimetuple ()
　　　　datetime. toordinal ()
　　　　datetime. weekday ()
　　　　datetime. isocalendar ()
　　　　datetime. isoformat ([ sep] )
　　　　datetime. ctime ()：返回一个日期时间的C格式字符串，等效于time.ctime(time.mktime(dt.timetuple()))；
　　　　datetime. strftime (format)

#2、timedelta类

#-------------------------计算相识时间-------------
def firstTimeToNow():
    now = datetime.datetime.now()
    first_time_tuple = datetime.datetime.strptime("2017/01/24","%Y/%m/%d")
    return (now - first_time_tuple).days

print("相见相识,第"+str(firstTimeToNow())+"天")