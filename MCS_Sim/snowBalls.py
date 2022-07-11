import pandas as pd
import numpy as np
from queue import Queue
from pprint import pprint
import os
from datetime import datetime
from time import time

class snowBalls:

    def __init__(self, type, year):

        self.data = None
        self.freez = 0          # 封闭期
        self.year = year
        self.type = type
        self.CLOSE = None
        self.KO = 1.03
        self.KI = 0.85
        self.coupon_monthly = 0.0044

        self._ParamDict = {
            '雪球模型' : type,
            '雪球时长': year,  # 雪球结构时长
            '敲出系数': 1,  # 敲出系数
            '敲入系数': 0.8,  # 敲入系数
        }


    def loadData(self, filename=''):

        path = os.getcwd() + '\\' + '000905.xlsx' + filename
        self.data = pd.read_excel(path, 'Sheet1')


    def obsv_date(self, i):

        pre, j = i, i+1     # 双指针
        if i > len(self.data) - 252 * self.year:
            print('over')
            return None
        FLAG = 0            # 出口标志
        count = 0           # 观察日计数器
        obsv_list = []      # 观察日列表
        base_day = self.data['Date'][pre].day    # 基准日
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
                if count >= self.freez and count <= 12*self.year-1 and month_now != pre_month:
                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count > 12*self.year-1:
                    FLAG = 1
                    # 出口
                    break
            elif day_now < day_pre and day_pre < base_day:
                if count >= self.freez and count <= 12*self.year-1 and month_now != pre_month:
                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][pre].month

            j += 1
            pre += 1
        return obsv_list


    def backTrader(self):

        res_dict = {
            'Date': [],
            'close' : [],
            'PE' : [],
            'realRes': [],
            'kickOUT_date' : [],
            'kickIN_date' : [],
            'out_price' : [],
            'profits' : [],
        }

        for i in range(len(self.data)):

            self.CLOSE = self.data['close'][i]
            obsv_list = self.obsv_date(i)
            if obsv_list:

                res = self.snowKick(obsv_list=obsv_list)

                res_dict['Date'].append(self.data['Date'][i])
                res_dict['close'].append(self.CLOSE)
                res_dict['PE'].append(self.data['pe_ttm'][i])
                res_dict['realRes'].append(res['FLAG'])
                res_dict['out_price'].append(res['out_price'])
                res_dict['profits'].append(res['profit'])
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

    def stat(self):

        stat_dict = {
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
            stat_dict['Date'].append(self.data['Date'][i].strftime('%Y-%m-%d'))
            stat_dict['close'].append(self.data['close'][i])
            stat_dict['kickOUT_date'].append(kickOUT_list)
            stat_dict['kickIN_date'].append(kickIN_list)
            stat_dict['profits'].append(profits)

        return pd.DataFrame(stat_dict)


    def snowKick(self, obsv_list:list, type='FCN'):
        ####################
        # 字典模拟 switch
        ####################
        res = self.FCN(obsv_list)

        return res


    def FCN(self, obsv_list:list):

        res = {
            'date' : None,          # 出场日期
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
            'lasting' : None,
            'obsv' : None,
        }
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                res['date'] = self.data['Date'][obsv]
                res['FLAG'] = 1
                res['out_price'] = self.data['close'][obsv]
                res['profit'] = (obsv_list.index(obsv) + 1) * self.coupon_monthly
                res['lasting'] = obsv_list.index(obsv) + 1
                res['obsv'] = obsv
                return res
        # 敲入
        if self.data['close'][obsv_list[-1]] < self.CLOSE * self. KI:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = -1
            res['profit'] = self.year * 12 * self.coupon_monthly - \
                            ((self.CLOSE * self. KI - self.data['close'][obsv_list[-1]] )/ self.CLOSE * self. KI)
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = len(obsv_list)
            res['obsv'] = obsv_list[-1]
            return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * 12 * self.coupon_monthly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = len(obsv_list)
            res['obsv'] = obsv_list[-1]
            return res

    def run(self):

        self.loadData()

        # df = self.backTrader()
        # df.to_excel('res.xlsx', index=False)

        df = self.stat()
        print('lenth:', len(df))
        pprint(df.head())
        df.to_excel('res.xlsx', index=False)

        return None

if __name__ == '__main__':
    print("启动模拟时间：", datetime.now())
    # 计时开始
    tic = time()
    # 初始化
    snowBall = snowBalls(type=None, year=1)
    # 运行蒙特卡洛模拟
    snowBall.run()
    # 计时结束
    tok = time()

    print("运行时间：", round(tok - tic, 2), 's')
