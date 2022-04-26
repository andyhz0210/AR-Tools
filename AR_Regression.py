# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:26:38 2022

@author: Andy Z Hu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import datetime
from scipy.special import comb
from itertools import combinations

#处理回归数据
def get_regression_data(start_date,end_date,x_list,y,x_raw,y_raw):
    #生成样本时间序列用来抓取回归数据
    date = pd.date_range(start_date,end_date,freq = "1Y")
    date_list = [datetime.datetime.strftime(i,'%F') for i in date]
    #抓取样本数据x,y
    x_sam = x_raw.loc[date_list][x_list]
    y_sam = y_raw.loc[date_list][y]
    #原数据x去量纲
    x_reg = x_sam/100+1
    #原数据y进行logit变换
    y_reg = np.log((y_sam/100)/(1-y_sam/100))
    return(x_reg,y_reg)

#计算VIF
def vifTest(ind_Candidate_df):
    X = sm.add_constant(ind_Candidate_df)
    viftest_list = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (viftest_list)

#多元回归
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
    result_df = result_df.reindex(columns=['Model','Dep', 'Coeff','P_value','R2', 'R2_adj','VIF', 'AIC']).head()
    return result_df

#遍历回归组合并返回计算结果
def BruteForceMultiCal(start_date, end_date, y, factor_number, x_raw, y_raw):
    possible_comb_num = comb(len(x_raw.columns), factor_number)
    print('共计 %d 种可能的因子组合将会被纳入遍历回归计算。' % possible_comb_num)
    possible_comb = list(combinations(x_raw.columns,factor_number))
    k=1
    all_result = np.empty((1,8))
    for i in possible_comb:
        model_name = 'Wilson'+ str(k)
        x_list = list(i)
        x_reg_temp,y_reg_temp = get_regression_data(start_date, end_date, x_list, y, x_raw, y_raw)
        result = np.array(multivariateLRTest(model_name,y_reg_temp, x_reg_temp))
        all_result = np.concatenate((all_result,result))
        k = k+1
    all_result = pd.DataFrame(all_result)
    all_result = all_result.drop([0])
    all_result.columns = ['Model','Dep', 'Coeff','P_value','R2', 'R2_adj','VIF', 'AIC']
    return all_result


#返回遍历最优解
def get_optimal_model(multiple_reg_result,factor_number):
    sorted_multiple_reg_result = multiple_reg_result.sort_values("R2_adj",inplace=False,ascending=False)
    optimal_list = []
    for i in range(factor_number+1):
        optimal_list.append(i)
    optimal_reg_result = sorted_multiple_reg_result.iloc[optimal_list,:]
    if factor_number == 3:
        x1 = optimal_reg_result.iloc[0,1]
        x1_coeff = optimal_reg_result.iloc[0,2]
        x2 = optimal_reg_result.iloc[1,1]
        x2_coeff = optimal_reg_result.iloc[1,2]
        x3 = optimal_reg_result.iloc[2,1]
        x3_coeff = optimal_reg_result.iloc[2,2]
        x4 = optimal_reg_result.iloc[3,1]
        x4_coeff = optimal_reg_result.iloc[3,2]
        x_list = []
        if x1 in list(x_raw.columns):
            x_list.append(x1)
        if x2 in list(x_raw.columns):
            x_list.append(x2)
        if x3 in list(x_raw.columns):
            x_list.append(x3)
        if x4 in list(x_raw.columns):
            x_list.append(x4)
        result = {}
        result[x1] = x1_coeff
        result[x2] = x2_coeff
        result[x3] = x3_coeff
        result[x4] = x4_coeff
    if factor_number == 2:
        x1 = optimal_reg_result.iloc[0,1]
        x1_coeff = optimal_reg_result.iloc[0,2]
        x2 = optimal_reg_result.iloc[1,1]
        x2_coeff = optimal_reg_result.iloc[1,2]
        x3 = optimal_reg_result.iloc[2,1]
        x3_coeff = optimal_reg_result.iloc[2,2]
        x_list = []
        if x1 in list(x_raw.columns):
            x_list.append(x1)
        if x2 in list(x_raw.columns):
            x_list.append(x2)
        if x3 in list(x_raw.columns):
            x_list.append(x3)
        result = {}
        result[x1] = x1_coeff
        result[x2] = x2_coeff
        result[x3] = x3_coeff
    x_reg,y_reg = get_regression_data(start_date, end_date, x_list, y, x_raw, y_raw)
    X = sm.add_constant(x_reg)
    regression = sm.OLS(y_reg,X)
    optimal_model = regression.fit()
    optimal_model.summary()
    return (optimal_model.summary(),result)


#读取原始数据（使用时请自行替换源文件地址）
x_raw = pd.read_excel('C:\\Users\\Andy Z Hu\\Desktop\\data_X.xlsx',header=0,index_col=0)
y_raw = pd.read_excel('C:\\Users\\Andy Z Hu\\Desktop\\data_Y.xlsx',header=0,index_col=0)


#获取入参
print('欢迎使用减值回归小工具！')
#入参1：样本开始日，样本结束日
start_date = datetime.datetime.strptime(input("请输入样本开始日："), '%Y/%m/%d')
end_date = datetime.datetime.strptime(input("请输入样本结束日："), '%Y/%m/%d')
#入参2：回归模型
factor_number = int(input("请输入回归因子数量："))
#入参3：行业：'农、林、牧、渔业'
y = input("请输入行业：")
#入参4：是否遍历
mode = input("普通回归请输入1，遍历回归请输入2：")
if mode == "1":
    #入参5：因子：'CPI:累计同比','社会消费品零售总额:累计同比'
    x_list = []
    while True:
        x = input("请输入因子名称，按'q'结束：")
        if x != 'q':
            x_list.append(x)
        if x == 'q':
            break
if mode =="2":
    pass


#执行简单回归（mode=1）
if mode == "1":
    #指定因子回归    
    x_reg, y_reg = get_regression_data(start_date, end_date, x_list, y, x_raw, y_raw)
    single_reg_result = multivariateLRTest('Wilson', y_reg, x_reg)
    single_reg_result.to_excel('指定因子回归结果.xlsx')


#执行遍历回归（mode=2）    
if mode == "2":
    #遍历回归
    multiple_reg_result = BruteForceMultiCal(start_date, end_date,y, factor_number,x_raw, y_raw)
    multiple_reg_result.to_excel('遍历回归结果.xlsx')
    #查询遍历最优解
    optimal_model, result = get_optimal_model(multiple_reg_result,factor_number)
    print(result)
    print(optimal_model)


#致谢
print("感谢使用，结果已经输出至代码所在文件夹。 by Andy")
