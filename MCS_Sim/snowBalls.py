import pandas as pd
import numpy as np
from queue import Queue
from pprint import pprint
import os
from datetime import datetime
from time import time

class snowBalls:

    def __init__(self, type:str):

        self.data = None
        self.type = type
        self.CLOSE = None
        self.showon = 0
        self.freez = 3          # 封闭期
        self.year = 1           # 雪球时长
        self.KO = 1.03          # 敲出系数
        self.KI = 0.85          # 敲入系数
        self.PE_style = 1       # PE计算方式：0为静态，1为滚动
        self.PE_static_thsh = 30       # 静态PE阈值，数值为PE绝对数值
        self.PE_trailing_thsh = 25        # 滚动PE阈值，数值为历史周期分位数
        self.PE_trailing_lenth = 252       # 滚动PE历史周期长度，数值单位：日
        self.coupon_monthly = 0.053 / 12   # 月票息
        self.N = 3 /100             # 大于年化 N% 概率阈值


    def loadData(self, filename=''):

        path = os.getcwd() + '\\' + '000905.xlsx' + filename
        self.data = pd.read_excel(path, 'Sheet1')


    def obsv_date(self, i):

        pre, j = i, i + 1  # 双指针
        if i > len(self.data) - 252 * self.year:
            print('over')
            return None
        FLAG = 0  # 出口标志
        count = 0  # 观察日计数器
        obsv_list = []  # 观察日列表
        base_day = self.data['Date'][pre].day  # 基准日
        pre_month = self.data['Date'][pre].month
        while 1:
            if FLAG:
                # 出口
                break
            try:
                day_now = self.data['Date'][j].day
                day_pre = self.data['Date'][pre].day
                month_now = self.data['Date'][j].month
            except:
                return obsv_list
            if day_now >= base_day:
                if count < self.freez and month_now != pre_month:
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count >= self.freez and count <= 12 * self.year - 1 and month_now != pre_month:
                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count > 12 * self.year - 1:
                    FLAG = 1
                    # 出口
                    break
            elif day_now < day_pre and day_pre < base_day:
                if count < self.freez and month_now != pre_month:
                    count += 1
                    pre_month = self.data['Date'][pre].month
                if count >= self.freez and count <= 12 * self.year - 1 and month_now != pre_month:
                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][pre].month

            j += 1
            pre += 1
        return obsv_list

    def get_PE_thsh(self, i):

        if self.PE_style == 0:
            return self.PE_static_thsh
        elif self.PE_style == 1:
            if i <= self.PE_trailing_lenth:
                return np.percentile(self.data['pe_ttm'][:i+1], self.PE_trailing_thsh)
            else:
                return np.percentile(self.data['pe_ttm'][i-self.PE_trailing_lenth : i+1], self.PE_trailing_thsh)


    def backTrader(self):

        res_dict = {
            'Date': [],
            'close' : [],
            'PE' : [],
            'realRes': [],
            'kickOUT_date' : [],
            'kickIN_date' : [],
            'out_price' : [],
            'lasting' : [],
            'profits' : [],
            'annually' : [],
        }

        for i in range(len(self.data)):
            thsh = 0
            if self.data['pe_ttm'][i] >= self.get_PE_thsh(i):
                thsh = 1  # PE高于阈值
            self.CLOSE = self.data['close'][i]
            obsv_list = self.obsv_date(i)
            if obsv_list:
                res = self.snowKick(obsv_list=obsv_list)
                if thsh:
                    if self.showon:
                        res_dict['Date'].append(self.data['Date'][i])
                        res_dict['close'].append(self.CLOSE)
                        res_dict['PE'].append(self.data['pe_ttm'][i])
                        res_dict['realRes'].append(None)
                        res_dict['out_price'].append(None)
                        res_dict['lasting'].append(None)
                        res_dict['profits'].append(None)
                        res_dict['annually'].append(None)
                        res_dict['kickOUT_date'].append(None)
                        res_dict['kickIN_date'].append(None)
                else:
                    res_dict['Date'].append(self.data['Date'][i])
                    res_dict['close'].append(self.CLOSE)
                    res_dict['PE'].append(self.data['pe_ttm'][i])
                    res_dict['realRes'].append(res['FLAG'])
                    res_dict['out_price'].append(res['out_price'])
                    res_dict['lasting'].append(res['lasting'])
                    res_dict['profits'].append(res['profit'])
                    annually = (1 + res['profit']) ** (12 / res['lasting']) - 1
                    res_dict['annually'].append(annually)
                    if res['FLAG'] == 1:
                        res_dict['kickOUT_date'].append(res['date'])
                        res_dict['kickIN_date'].append(np.nan)
                    elif res['FLAG'] == 0:
                        res_dict['kickOUT_date'].append(np.nan)
                        res_dict['kickIN_date'].append(np.nan)
                    elif res['FLAG'] == -1:
                        res_dict['kickOUT_date'].append(np.nan)
                        res_dict['kickIN_date'].append(res['date'])
            else:
                break

        res_df = pd.DataFrame(res_dict)

        return res_df

    def Trailing(self):
        """
        滚动雪球
        :return:
        """
        rolling_dict = {
            'Date': [],
            'close': [],
            'kickOUT_date': [],
            'kickIN_date': [],
            'profits': [],
        }

        for i in range(len(self.data)):
            self.CLOSE = self.data['close'][i]
            p = i
            lasted = 0
            profits = 0
            kickOUT_list = []
            kickIN_list = []
            while 1:
                if lasted >= 12:
                    break
                obsv_list = self.obsv_date(p)
                if obsv_list:
                    res = self.snowKick(obsv_list)
                    profits += res['profit']
                    if res['FLAG'] == 1:
                        kickOUT_list.append(res['date'].strftime('%Y-%m-%d'))
                    else:
                        kickIN_list.append(res['date'].strftime('%Y-%m-%d'))
                    lasted += res['lasting']
                else:
                    break
                p = res['obsv'] + 1
            rolling_dict['Date'].append(self.data['Date'][i].strftime('%Y-%m-%d'))
            rolling_dict['close'].append(self.data['close'][i])
            rolling_dict['kickOUT_date'].append(kickOUT_list)
            rolling_dict['kickIN_date'].append(kickIN_list)
            rolling_dict['profits'].append(profits)

        return pd.DataFrame(rolling_dict)


    def stat(self, df:pd.DataFrame):

        stat_dict = {
            '分类' : ['敲入', '', '敲出', '稳定', '平均敲出月份数', '总计', '亏损', '', '', '', '', '', '盈利', '', '', '', '', '', '', '年化收益>{}%的概率'.format(self.N*100), '绝对收益>{}%的概率'.format(self.N*100)],
            '情形' : ['盈利', '亏损', '盈利', '盈利', '', '', '亏损最大值', '亏损1/4', '亏损2/4', '亏损3/4', '亏损最小值', '平均数', '盈利最大值', '盈利1/4', '盈利2/4', '盈利3/4', '盈利最小值', '平均数', '', '', ''],
            '次数' : ['' for _ in range(21)],
            '平均收益/亏损' : ['' for _ in range(21)],
            '不加票息' : ['' for _ in range(21)],
        }
        profits_kickOUT = [df['profits'][i] for i in range(len(df))
                                            if df['realRes'][i] == 1]
        profits_kickIN = [df['profits'][i] for i in range(len(df))
                                            if df['realRes'][i] == -1]
        profits_stable = [df['profits'][i] for i in range(len(df))
                          if df['realRes'][i] == 0]
        WIN = [df['profits'][i] for i in range(len(df))
                                if df['profits'][i] > 0]
        LOST = [df['profits'][i] for i in range(len(df))
                                if df['profits'][i] <= 0]
        abs_profit_len = sum(profit > self.N for profit in df['profits'])
        annually_profit_len = sum(annual > self.N for annual in df['annually'])
        stat_dict['次数'][0] = (sum(profit > 0 for profit in profits_kickIN))
        stat_dict['次数'][1] = (sum(profit <= 0 for profit in profits_kickIN))
        stat_dict['次数'][2] = (sum(profit > 0 for profit in profits_kickOUT))
        stat_dict['次数'][3] = (sum(profit > 0 for profit in profits_stable))
        stat_dict['次数'][5] = sum(stat_dict['次数'][:4])

        stat_dict['次数'][6] = (min(LOST))
        stat_dict['次数'][7] = (np.nanpercentile(LOST, 75))
        stat_dict['次数'][8] = (np.nanpercentile(LOST, 50))
        stat_dict['次数'][9] = (np.nanpercentile(LOST, 25))
        stat_dict['次数'][10] = (max(LOST))
        stat_dict['次数'][11] = (np.nanmean(LOST))
        stat_dict['次数'][12] = (max(WIN))
        stat_dict['次数'][13] = (np.nanpercentile(WIN, 25))
        stat_dict['次数'][14] = (np.nanpercentile(WIN, 50))
        stat_dict['次数'][15] = (np.nanpercentile(WIN, 75))
        stat_dict['次数'][16] = (min(WIN))
        stat_dict['次数'][17] = (np.nanmean(WIN))

        stat_dict['次数'][19] = (annually_profit_len / len(df))
        stat_dict['次数'][20] = (abs_profit_len / len(df))

        stat_dict['平均收益/亏损'][0] = np.nanmean([profit for profit in profits_kickIN if profit > 0])
        stat_dict['平均收益/亏损'][1] = np.nanmean([profit for profit in profits_kickIN if profit <= 0])
        stat_dict['平均收益/亏损'][2] = np.nanmean([profit for profit in profits_kickOUT])
        stat_dict['平均收益/亏损'][3] = np.nanmean([profit for profit in profits_stable])


        # 平均敲出月份数
        stat_dict['情形'][4] = np.nanmean([df['lasting'][i] for i in range(len(df)) if df['realRes'][i] == 1])

        stat_dict['不加票息'][0] = np.nanmean([profit - self.coupon_monthly * self.year * 12 for profit in profits_kickIN if profit > 0])
        stat_dict['不加票息'][1] = np.nanmean([profit - self.coupon_monthly * self.year * 12 for profit in profits_kickIN if profit <= 0])

        return pd.DataFrame(stat_dict)


    def snowKick(self, obsv_list:list, type='FCN'):
        ####################
        # 字典模拟 switch
        ####################
        res = self.FCN(obsv_list)

        return res


    def FCN(self, obsv_list: list):

        res = {
            'date': None,  # 出场日期
            'FLAG': None,
            'profit': None,
            'out_price': None,
            'lasting': None,
            'obsv': None,
        }
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                res['date'] = self.data['Date'][obsv]
                res['FLAG'] = 1
                res['out_price'] = self.data['close'][obsv]
                res['profit'] = (self.freez + obsv_list.index(obsv) + 1) * self.coupon_monthly
                res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                res['obsv'] = obsv
                return res
        # 敲入
        if self.data['close'][obsv_list[-1]] < self.CLOSE * self.KI:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = -1
            res['profit'] = self.year * 12 * self.coupon_monthly - \
                            ((self.CLOSE * self.KI - self.data['close'][obsv_list[-1]]) / self.CLOSE)
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
            res['obsv'] = obsv_list[-1]
            return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * 12 * self.coupon_monthly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
            res['obsv'] = obsv_list[-1]
            return res

    def run(self):

        self.loadData()

        filename = self.type + '_res_' + str(datetime.now().hour) + '-' +  str(datetime.now().minute) + '.xlsx'
        writer = pd.ExcelWriter(filename)

        df = self.backTrader()

        df_stat = self.stat(df)

        # print('lenth:', len(df))
        # pprint(df.head())
        # df.to_excel('res.xlsx', 'Sheet2', index=False)

        self.showon = 1
        df_full = self.backTrader()

        df_full.to_excel(writer, 'Sheet1', index=False)
        df_stat.to_excel(writer, 'Sheet2', index=False)
        # 保存
        writer.save()

        return None

if __name__ == '__main__':
    print("启动回测时间：", datetime.now())
    # 计时开始
    tic = time()
    # 初始化
    snowBall = snowBalls(type='FCN')
    # 运行回测
    snowBall.run()
    # 计时结束
    tok = time()

    print("运行时间：", round(tok - tic, 2), 's')
