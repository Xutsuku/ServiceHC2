
# -*- coding: utf-8 -*-

from elangc.hourErlangc import calSL
import pandas as pd
from pandas import read_excel

#
# # data = pd.read_clipboard()
# '''   # data excel
# hour	1小时呼入量	平均通话时长
# 0	20	447
# 1	14	447
# 2	7	447
# '''
path ="d:\\Users\\lfzhou\\Desktop\\服务运营HC 流程化\\ErlangC.xlsx"
data = read_excel(path,sheetname='ErlangC')

waittime = 20
sl = 80
result = calSL(data, waittime, sl)
print(result)


