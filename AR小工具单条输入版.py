# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.special import comb
from itertools import combinations

#从源数据中提取样本期间内回归数据
def get_regression_data(start_date,end_date,x_list,y,x_raw,y_raw):
    #定义样本期间
    date = pd.date_range(start_date,end_date,freq = "1Y")
    date_list = [datetime.datetime.strftime(i,'%F') for i in date]
    #抓取样本期间内源数据
    x_sam = x_raw.loc[date_list][x_list]
    y_sam = y_raw.loc[date_list][y]
    #宏观因子x去量纲
    x_reg = x_sam/100+1
    #行业不良率y进行logit变换
    y_reg = np.log((y_sam/100)/(1-y_sam/100))
    return(x_reg,y_reg)

#VIF检验：测试回归是否存在多重共线性
def vifTest(x_reg):
    x_reg_adj = sm.add_constant(x_reg)
    viftest_list = [variance_inflation_factor(x_reg_adj.values, i) for i in range(x_reg_adj.shape[1])]
    return (viftest_list)

#多元回归
def multivariateLRTest(model_name,y_reg, x_reg):
    # 进行VIF检验
    vif = vifTest(x_reg)
    # 进行多元回归
    x_reg_adj = sm.add_constant(x_reg)
    regression = sm.OLS(y_reg, x_reg_adj)
    LR = regression.fit()
    # 统计回归结果
    LR_coef = list(LR.params)
    LR_R2 = LR.rsquared
    LR_R2_adj = LR.rsquared_adj
    LR_PValue = list(LR.pvalues)
    LR_AIC = LR.aic
    indep = ['Intercept'] + x_reg.columns.tolist()
    reg_result = pd.DataFrame({'Model':model_name,'Indep':indep,'Coeff': LR_coef,'P_value': LR_PValue,'R2': LR_R2,'R2_adj':LR_R2_adj,'VIF':vif,'AIC':LR_AIC})
    return LR.summary(),reg_result

#遍历回归组合并返回计算结果
def BruteForceMultiCal(start_date, end_date, y, factor_number, x_raw, y_raw):
    possible_comb_num = comb(len(x_raw.columns), factor_number)
    print('共计 %d 种可能的因子组合将会被纳入遍历回归计算。' % possible_comb_num)
    possible_comb = list(combinations(x_raw.columns,factor_number))
    k=1
    comb_result = np.empty((1,8))
    for i in possible_comb:
        model_name = 'Comb'+ str(k)
        x_list = list(i)
        x_reg_temp,y_reg_temp = get_regression_data(start_date, end_date, x_list, y, x_raw, y_raw)
        LR_summary,LR_result = multivariateLRTest(model_name,y_reg_temp, x_reg_temp)
        reg_result = np.array(LR_result)
        comb_result = np.concatenate((comb_result,reg_result))
        k = k+1
    comb_result = pd.DataFrame(comb_result)
    comb_result = comb_result.drop([0])
    comb_result.columns = ['Model','Indep', 'Coeff','P_value','R2', 'R2_adj','VIF', 'AIC']
    return comb_result

#返回遍历最优解
def get_optimal_model(comb_result,factor_number,x_raw):
    sorted_comb_result = comb_result.sort_values("R2_adj",inplace=False,ascending=False)
    optimal_index = np.arange(factor_number+1).tolist()
    optimal_reg_result = sorted_comb_result.iloc[optimal_index,:]
    optimal_x_list = []
    optimal_result = {}
    for i in optimal_index:
        x = 'x' + str(i)
        x = optimal_reg_result.iloc[i,1]
        x_coeff = optimal_reg_result.iloc[i,2]
        optimal_result[x] = x_coeff
        if x in list(x_raw.columns):
            optimal_x_list.append(x)
    return(optimal_x_list,optimal_result)

#计算宏观因子预测值与真实值差值的标准差
def get_std(x_base_pred,x_real,start_date,end_date):
    #生成样本期间事件序列
    date = pd.date_range(start_date,end_date,freq = "1Y")
    date_list = [datetime.datetime.strftime(i,'%F') for i in date]
    #抓取样本数据
    x_real_sam = x_real.loc[date_list][:]
    x_base_pred_sam = x_base_pred.loc[date_list][:]
    #计算所有宏观因子预测值和真实值之差
    x_diff_sam = x_real_sam - x_base_pred_sam
    #计算所有宏观因子预测值与真实值差值的标准差，输出PD格式表
    diff_std = []
    for i in x_diff_sam.columns:
        diff_std.append(x_diff_sam[i].std())
    pd_diff_std = pd.DataFrame(diff_std,index=x_diff_sam.columns,columns=['Standard Deviation'])
    return pd_diff_std

#计算乐观、悲观情况下宏观因子的预测值
def get_x_pred(x_name,result,opt_wgt,pes_wgt,x_pred_raw,pd_x_std,pred_date):
    #计算乐观、悲观分位点
    opt_quantile = stats.norm(0,1).ppf(opt_wgt/2)
    pes_quantile = stats.norm(0,1).ppf(pes_wgt/2)
    x_coeff = result[x_name]
    x_pred_raw_temp = x_pred_raw.loc[pred_date][:]
    x_pred_raw_temp.index = list(x_pred_raw_temp.loc[:]['宏观因子'])
    x_pred_raw_temp = x_pred_raw_temp.drop('宏观因子',axis=1)
    base_x_pred = x_pred_raw_temp.loc[x_name]['预测值']
    x_std = pd_x_std.loc[x_name]['Standard Deviation']
    if x_coeff > 0:
        opt_x_pred = base_x_pred + opt_quantile * x_std
        pes_x_pred = base_x_pred - pes_quantile * x_std
    if x_coeff < 0:
        opt_x_pred = base_x_pred - opt_quantile * x_std
        pes_x_pred = base_x_pred + pes_quantile * x_std
    return ([opt_x_pred,base_x_pred,pes_x_pred])

#计算乐观、基准、悲观情况下的前瞻后不良率
def get_pred_y(reg_result,pd_pred_result):
    x_list = list(pd_pred_result.index)
    opt_logit_y = 0
    base_logit_y = 0
    pes_logit_y = 0
    for i in x_list:
        opt_logit_y = opt_logit_y + pd_pred_result.loc[i]["乐观预测值"] * reg_result[i]
        base_logit_y = base_logit_y + pd_pred_result.loc[i]["基准预测值"] * reg_result[i]
        pes_logit_y = pes_logit_y + pd_pred_result.loc[i]["悲观预测值"] * reg_result[i]
    opt_logit_y = opt_logit_y + reg_result["Intercept"]
    base_logit_y = base_logit_y + reg_result["Intercept"]
    pes_logit_y = pes_logit_y + reg_result["Intercept"]
    opt_y = np.exp(opt_logit_y)/(1+np.exp(opt_logit_y))*100
    base_y = np.exp(base_logit_y)/(1+np.exp(base_logit_y))*100
    pes_y = np.exp(pes_logit_y)/(1+np.exp(pes_logit_y))*100
    return opt_y, base_y, pes_y

#计算加权前瞻后不良率
def get_wgt_pred_y(opt_y, base_y, pes_y,opt_wgt,base_wgt,pes_wgt):
    wgt_pred_y = opt_y*opt_wgt + base_y*base_wgt + pes_y*pes_wgt
    return wgt_pred_y


# 使用须知
print("以下为使用须知：")
print("1.请将‘AR小工具’文件夹拷贝至C盘根目录;")
print("欢迎使用！")

# 读取宏观因子实际值（Wind）
x_real = pd.read_excel('/Users/andy/Desktop/AR小工具/源数据/宏观因子实际值（Wind）.xlsx', header=0, index_col=0)
# 读取行业历史不良率（Wind）
y_real = pd.read_excel('/Users/andy/Desktop/AR小工具/源数据/行业历史不良率（Wind）.xlsx', header=0, index_col=0)
# 读取宏观因子预测值（Wind）
x_base_pred = pd.read_excel('/Users/andy/Desktop/AR小工具/源数据/宏观因子预测值（Wind）.xlsx', header=0, index_col=0)
# 读取宏观因子基准预测值
x_pred_raw = pd.read_excel('/Users/andy/Desktop/AR小工具/源数据/宏观因子基准预测值.xlsx', header=0, index_col=0)
# 读取行业乐基悲权重
wgt = pd.read_excel('/Users/andy/Desktop/AR小工具/源数据/行业乐基悲权重.xlsx', header=0, index_col=0)

print('欢迎使用AR小工具！请填写以下参数：\n')
# 入参1：样本开始日，样本结束日
start_date = datetime.datetime.strptime(input("请输入样本开始日："), '%Y/%m/%d')
end_date = datetime.datetime.strptime(input("请输入样本结束日："), '%Y/%m/%d')
# 入参2：因子数量
factor_number = int(input("请输入回归因子数量："))
# 入参3：行业：'农、林、牧、渔业'
y = input("请输入行业：")
# 入参4：选择回归模型：指定因子进行回归/遍历回归
mode = input("普通回归请输入1，遍历回归请输入2：")
if mode == "1":
    # 入参5：因子：'CPI:累计同比','社会消费品零售总额:累计同比'
    x_list = []
    while True:
        x = input("请输入因子名称，按'q'结束：")
        if x != 'q':
            x_list.append(x)
        if x == 'q':
            break
if mode == "2":
    pass

#入参5：前瞻预测日期
pred_date = input("请输入前瞻预测日期：")

#回归运算
if mode == "1":
    # 指定因子回归
    x_reg, y_reg = get_regression_data(start_date, end_date, x_list, y, x_real, y_real)
    LR_summary, LR_result = multivariateLRTest('Wilson', y_reg, x_reg)
    file_name = "/Users/andy/Desktop/AR小工具/Result/指定因子回归结果.xlsx"
    LR_result.to_excel(file_name)
    print(LR_summary)
    print('\n')
    x_list, result_dict = get_optimal_model(LR_result, factor_number, x_real)

if mode == "2":
    # 遍历回归
    comb_result = BruteForceMultiCal(start_date, end_date, y, factor_number, x_real, y_real)
    y_temp = y.replace(':','：')
    file_name = "/Users/andy/Desktop/AR小工具/Result/" + y_temp + "-遍历回归结果.xlsx"
    comb_result.to_excel(file_name)
    # 返回并打印遍历最优解
    optimal_x_list, result_dict = get_optimal_model(comb_result, factor_number, x_real)
    x_reg, y_reg = get_regression_data(start_date, end_date, optimal_x_list, y, x_real, y_real)
    LR_summary, LR_result = multivariateLRTest('Wilson', y_reg, x_reg)
    LR_summary_text = LR_summary.as_text()
    file_name = "/Users/andy/Desktop/AR小工具/Result/" + y_temp + "-最优回归结果.csv"
    LR_summaryFile = open(file_name, 'w')
    LR_summaryFile.write(LR_summary_text)
    LR_summaryFile.close()
    print(LR_summary)
    print('\n')

# 根据行业获取对应乐基悲权重
opt_wgt = wgt.loc[y]['乐观权重']/100
base_wgt = wgt.loc[y]['正常权重']/100
pes_wgt = wgt.loc[y]['悲观权重']/100
print('乐观权重：',opt_wgt)
print('基准权重：',base_wgt)
print('悲观权重：',pes_wgt)

# 计算各宏观因子实际值与预测值偏离的标准差
x_std_df = get_std(x_base_pred, x_real, start_date, end_date)
print("\n宏观因子在样本期间内预测值与实际值偏离的标准差结果如下：\n")
print(x_std_df,'\n')

# 返回包含所有宏观因子乐观、基准、悲观预测值的PD表
x_list = []
for key in result_dict:
    x_list.append(key)
x_list.remove('Intercept')
pred_result = []
for x in x_list:
    pred_result.append(get_x_pred(x, result_dict, opt_wgt, pes_wgt, x_pred_raw, x_std_df,pred_date))
pred_result_df = pd.DataFrame(pred_result, index=x_list, columns=['乐观预测值', '基准预测值', '悲观预测值'])
file_name = "/Users/andy/Desktop/AR小工具/Result/" + y_temp + "-宏观因子乐基悲预测值.xlsx"
pred_result_df.to_excel(file_name)
pred_result_adj_df = pred_result_df / 100 + 1

opt_y, base_y, pes_y = get_pred_y(result_dict, pred_result_adj_df)
y_pred_result_df = pd.DataFrame(get_pred_y(result_dict, pred_result_adj_df), columns=["前瞻后结果"],
                                index=["前瞻后乐观不良率", "前瞻后基准不良率", "前瞻后悲观不良率"])
file_name = "/Users/andy/Desktop/AR小工具/Result/" + y_temp + '-前瞻后乐基悲不良率.xlsx'
y_pred_result_df.to_excel(file_name)
print(y_pred_result_df,'\n')

# 加权前瞻后不良率
wgt_pred_y = get_wgt_pred_y(opt_y, base_y, pes_y, opt_wgt, base_wgt, pes_wgt)
print("加权前瞻后不良率：" + str(wgt_pred_y),'\n')

# 样本期间内行业历史平均不良率求平均
date = pd.date_range(start_date, end_date, freq="1Y")
date_list = [datetime.datetime.strftime(i, '%F') for i in date]
his_y_df = y_real.loc[date_list][y]
his_y_mean = his_y_df.mean()
print(y+"在样本期间内的历史不良率平均值：",his_y_mean,'\n')

# 计算前瞻后倍数
multiples = wgt_pred_y / his_y_mean
print("前瞻倍数：" + str(multiples),'\n')

input("please input any key to exit!")