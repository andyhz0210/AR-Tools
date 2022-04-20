# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:26:38 2022

@author: Andy Z Hu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy import stats
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
import datetime


#入参1：样本开始日，样本结束日
start_date = datetime.datetime.strptime(input("请输入样本开始日："), '%Y/%m/%d')
end_date = datetime.datetime.strptime(input("请输入样本结束日："), '%Y/%m/%d')

#入参2：回归模型
regression_mode = input("请输入回归模型：双因子/三因子：")

#入参3：因子：'CPI:累计同比','社会消费品零售总额:累计同比'
x_list = []
while True:
    x = input("请输入因子名称，按'q'结束：")
    if x != 'q':
        x_list.append(x)
    if x == 'q':
        break
    
#入参4：行业：'农、林、牧、渔业'
y_list = [input("请输入行业：")]

#生成样本时间序列用来抓取回归数据
date = pd.date_range(start_date,end_date,freq = "1Y")
date_list = [datetime.datetime.strftime(i,'%F') for i in date]

#读取原数据x,y
data_x = pd.read_excel('C:\\Users\\Andy Z Hu\\Desktop\\data_X.xlsx',header=0,index_col=0)
data_y = pd.read_excel('C:\\Users\\Andy Z Hu\\Desktop\\data_Y.xlsx',header=0,index_col=0)
x_test = np.array(data_x.loc[date_list][x_list])
y_test = np.array(data_y.loc[date_list][y_list])

#原数据x去量纲
for i in x_test:
    i[0] = i[0]/100+1
    i[1] = i[1]/100+1
    
#原数据y进行logit变换
for i in y_test:
    i[0] = math.log((i[0]/100)/(1-i[0]/100),math.e)

#回归数据转换至PD表    
df_x = pd.DataFrame(x_test)
df_x.index = date_list
df_x.columns = x_list
2009
df_y = pd.DataFrame(y_test)
df_y.index = date_list
df_y.columns = y_list


def vifTest(ind_Candidate_df):
    X = sm.add_constant(ind_Candidate_df)
    viftest_list = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (viftest_list)

def multivariateLRTest(model_name,dep_t, ind_raw):
    # 进行VIF检验
    vif = vifTest(ind_raw)
    # 进行线性回归
    X = sm.add_constant(ind_raw)
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    # params
    LR_coef = list(LR.params)
    LR_R2 = LR.rsquared
    LR_R2_adj = LR.rsquared_adj
    LR_PValue = list(LR.pvalues)
    LR_AIC = LR.aic
    result_df = pd.DataFrame({'VIF':vif,'Coeff': LR_coef, 'P_value': LR_PValue, 'R2': LR_R2,'R2_adj':LR_R2_adj,'AIC':LR_AIC})
    result_df['Model'] = model_name
   
    intercept=['Intercept']+ind_raw.columns.tolist()
    result_df['Dep'] = intercept
    return result_df

result = multivariateLRTest('Wilson',df_y, df_x)
pd_result = pd.DataFrame(result)
print(result)