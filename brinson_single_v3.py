# -*- coding: utf8 -*-
# author: Xu Huanrong
# single period

import seaborn as sns
from WindPy import w
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd

plt.rcParams['font.sans-serif'] = ['SimHei']  # display Chinese Character
plt.rcParams['axes.unicode_minus'] = False  # display positive/negative mathematical symbols


def valuation_data_read(filename):
    "reading valuation table and return list of rows"
    data = xlrd.open_workbook(filename)  #privacy preserved
    sheet = data.sheets()[0]
    nrows = sheet.nrows
    ncols = sheet.ncols
    data_list = []

    for i in range(2, nrows):
        row_data = sheet.row_values(i) 
        data_list.append(row_data)
    
    return data_list
    
def data_fliter(data_list):
    "Process raw data and return stock list"
    temp_list = []

    for row_data in data_list:
        if row_data[11] != ' ' and row_data[11] != '': 
            temp_list.append(row_data)

    return temp_list

def to_wind_code(raw_data):
    "change raw code to wind code."
    if raw_data.startswith('6'):
        stock_code = '{0}.{1}'.format(raw_data, 'SH') # reformat to Wind stock code
    else:
        stock_code = '{0}.{1}'.format(raw_data, 'SZ') # reformat to Wind stock code
    return stock_code
    
def std_data(temp_list):
    "Return standard data format."
    stocks_list = []
    weight_num = 0.0
    for row_data in temp_list:
        raw_data = str(row_data[1])[8:14]
        stock_code = to_wind_code(raw_data)
        stock_name = row_data[2]
        w.start()
        wsd_data = w.wsd(stock_code, "industry_sw", "2017-12-29", "2017-12-29", "industryType=1")
        stock_industry = wsd_data.Data[0][0]
        cost = row_data[5]
        weight = row_data[6]
        weight_num += weight
        fund_yield_rate = row_data[8] / cost - 1
        stock_list = [stock_code, stock_name, stock_industry, weight, fund_yield_rate]
        stocks_list.append(stock_list)
        stocks_df = pd.DataFrame(stocks_list, columns=['stock_code', 'stock_name', 'stock_industry', 'weight', 'fund_yield_rate'])
        stocks_df['weight'] = stocks_df['weight']
        new_rows = ['000000.XJ', '现金资产', '现金', 1 - weight_num, 0.00]
        stocks_df.loc[stocks_df.shape[0]] = new_rows
    return stocks_df
    
# get benchmark info: "date=2017-12-29;windcode=000300.SH"
def benchmark_data_read(filename):
    "reading benchmark table and return list of rows"
    data = xlrd.open_workbook(filename)  #privacy preserved
    sheet = data.sheets()[0]
    nrows = sheet.nrows
    ncols = sheet.ncols
    bench_list = []
    
    for i in range(1, nrows):
        industry =  sheet.row_values(i)[16]
        weight = sheet.row_values(i)[4]
        day_yield = sheet.row_values(i)[6]
        temp_list = [industry, weight, day_yield]
        bench_list.append(temp_list)
    
    bench_df = pd.DataFrame(bench_list, columns=['stock_industry', 'weight', 'fund_yield_rate'])
    bench_df['weight'] = bench_df['weight'] * 0.5 / 100
    bench_df['fund_yield_rate'] = bench_df['fund_yield_rate'] / 100
    new_rows = ['现金', 0.5, 0.00] # cash weights 50% in benchmark
    bench_df.loc[bench_df.shape[0]] = new_rows
    return bench_df

def result_df(dframe):    
    "Return fund weight and yiled."
    dframe['yield'] = dframe['weight'] * dframe['fund_yield_rate']
    weight_df = dframe.groupby('stock_industry')['weight'].sum()
    yield_df = dframe.groupby('stock_industry')['yield'].sum()
    weight_df = weight_df.to_frame()
    yield_df = yield_df.to_frame()
    weight_df['industry'] = weight_df.index
    yield_df['industry'] = weight_df.index
    weight_df =  weight_df.reset_index(drop= True)
    yield_df =  yield_df.reset_index(drop= True)
    yield_df['yield'] = yield_df['yield'] / weight_df['weight']
    return weight_df, yield_df

    
def to_all_key(df_1, df_2, key):
    "change industry numbers of fund as benchmark's "

    df = df_1.copy()
    df.loc[:, key] = np.array([0.0] * len(df.industry))
    new_df = df_2.append(df, ignore_index=True)
    new_df = new_df.drop_duplicates(subset='industry')
    return new_df
    
def re_index(df):
    "Reindex the df passed"
    df = df.sort_values(by=['industry'])
    df = df.set_index(['industry'])
    return df

def draw_graph(result1, result2, result3, result4):
    sns.set_color_codes("muted")
    result_set = [[result1, 'allocation', '行业配置收益'], 
                  [result2, 'selection', '个股选择收益'],
                  [result3, 'interaction', '综合收益']]
    # drawing
    for result in result_set:
        pd_df = result[0].sort_values("result", ascending=False)
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(pd_df.result, pd_df.index, color="b")
        ax.set(xlabel=result[1], ylabel="industry")
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(round(x, 6))))
        ax.set_yticklabels(pd_df.index)
        ax.set(ylabel="", xlabel=result[2])
        plt.tight_layout()
        plt.savefig("brinson_model_" + result[1] + ".png", dpi=120)
    
    # draw result bar.
    index = ['行业配置', '个股选择', '交叉影响']
    pd_df = pd.DataFrame(result4, index=index, columns=['yield'])
    pd_df['xlabel'] = [1, 2, 3]
    plt.figure(figsize=(10, 8))
    plt.bar(pd_df['xlabel'], pd_df['yield'], tick_label=pd_df.index,data=True)
    for a,b in zip(pd_df['xlabel'], pd_df['yield']):    
        plt.text(a, b+0.00005, str(round(b, 6) * 100) + '%', ha='center', va='center', fontsize=11)    
    plt.savefig("brinson_model_summary.png", dpi=120)
    
def get_results(fund_weight_df, fund_yield_df, bench_weight_df, bench_yield_df):
    "Produce result and graph."
    
    bench_weight_df = re_index(bench_weight_df)
    bench_yield_df = re_index(bench_yield_df)
    fund_weight_df = re_index(fund_weight_df)
    fund_yield_df = re_index(fund_yield_df)
        
        
    allocation_result =  (fund_weight_df['weight'] - bench_weight_df['weight']) * bench_yield_df['yield']
    allocation_df =  allocation_result.to_frame(name='result')
    industry_allocation = allocation_df.sum()[0]
    
    selection_result = (fund_yield_df['yield'] - bench_yield_df['yield']) * bench_weight_df['weight']
    selection_df = selection_result.to_frame(name='result')
    stock_selection = selection_df.sum()[0]

    interaction_result = (fund_weight_df['weight'] - bench_weight_df['weight']) * (fund_yield_df['yield'] - bench_yield_df['yield'])
    interaction_df = interaction_result.to_frame(name='result')
    interaction = interaction_df.sum()[0]

    result_division = [industry_allocation, stock_selection, interaction]
    draw_graph(allocation_df, selection_df, interaction_df, result_division)
    
    return result_division

if __name__ == '__main__':
    filename1 = 'valuation.xls'
    filename = 'benchmark.xlsx'
    w.start()
    data_list = valuation_data_read(filename1)
    temp_list = data_fliter(data_list)
    stocks_df = std_data(temp_list)
    bench_df = benchmark_data_read(filename)
    fund_weight_df, fund_yield_df= result_df(stocks_df)
    bench_weight_df, bench_yield_df = result_df(bench_df)
    fund_weight_df = to_all_key(bench_weight_df, fund_weight_df, 'weight')
    fund_yield_df = to_all_key(bench_yield_df, fund_yield_df, 'yield')
    result_division= get_results(fund_weight_df, fund_yield_df, bench_weight_df, bench_yield_df)
    print(result_division)