import pandas as pd
import numpy as np
from queue import Queue
from pprint import pprint
import os
from datetime import datetime
from time import time

class snowBalls:

    def __init__(self, type, year, coupon_yearly, freez=0, KO = 1.03, KI = 0.75):
        self.N = 3 /100             # 大于年化 N% 概率阈值
        self.data = None
        self.freez = freez          # 封闭期
        self.year = year
        self.type = type
        self.CLOSE = None
        self.KO = KO
        self.KI = KI
        self.coupon_yearly = coupon_yearly
        self.coupon_monthly = coupon_yearly/12
        self.i = None
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
                if count < self.freez and month_now != pre_month:
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count >= self.freez and count <= 12*self.year-1 and month_now != pre_month:
                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count > 12*self.year-1:
                    FLAG = 1
                    # 出口
                    break
            elif day_now < day_pre and day_pre < base_day:
                if count < self.freez and month_now != pre_month:
                    count += 1
                    pre_month = self.data['Date'][pre].month
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
            self.i = i
            self.CLOSE = self.data['close'][i]
            obsv_list = self.obsv_date(i)
            if obsv_list:

                res = self.snowKick(obsv_list=obsv_list,type = self.type)

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
        if type == 'FCN':
            res = self.FCN(obsv_list)
        if type == 'OTM':
            res = self.OTM(obsv_list)
        if type == 'SSS':
            res = self.SSS(obsv_list)
        if type == 'DSS':
            res = self.DSS(obsv_list)
        if type == 'CLA':
            res = self.CLA(obsv_list)
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
                res['profit'] = (self.freez + obsv_list.index(obsv) + 1) * self.coupon_monthly
                res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                res['obsv'] = obsv
                return res
        # 敲入
        if self.data['close'][obsv_list[-1]] < self.CLOSE * self. KI:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = -1
            res['profit'] = self.year * 12 * self.coupon_monthly - \
                            ((self.CLOSE * self. KI - self.data['close'][obsv_list[-1]] )/ self.CLOSE)
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

    # def run(self):

    #     self.loadData()

    #     df = self.backTrader()
    #     df.to_excel('res.xlsx', index=False)

    #     # df = self.stat()
    #     # print('lenth:', len(df))
    #     # pprint(df.head())
    #     # df.to_excel('res.xlsx', index=False)


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
    
    
    def OTM(self, obsv_list:list):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
        }
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                res['date'] = self.data['Date'][obsv]
                res['FLAG'] = 1
                res['out_price'] = self.data['close'][obsv]
                res['profit'] = res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
                return res
        # 敲入
        price_min = np.min(self.data['close'][i:i+252])
         
        
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE * self. KI:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE * self. KI - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_yearly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            return res


    # 敲出价递减（单降）
    def SSS(self, obsv_list:list, KO_decrease_months=6, down=0.005):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
        }
        # 敲出
        for obsv in obsv_list:
            if (self.freez + obsv_list.index(obsv)) < KO_decrease_months:                
                if self.data['close'][obsv] >= self.CLOSE * self.KO:
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    return res
            else:
                sum_down = down *((self.freez + obsv_list.index(obsv)) - KO_decrease_months)
                if self.data['close'][obsv] >= self.CLOSE * (self.KO - sum_down):
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    return res
        # 敲入
        price_min = np.min(self.data['close'][i:i+252])
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_yearly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            return res


    # 敲出价、收益率递减（双降）
    def DSS(self, obsv_list:list, KO_decrease_months=6, down_price=0.005, down_ret=0.01):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': float(0),
            'out_price' : None,
        }
        # 敲出
        for obsv in obsv_list:
            if (self.freez + obsv_list.index(obsv)) < KO_decrease_months:                
                if self.data['close'][obsv] >= self.CLOSE * self.KO:
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    for obsv_mouths in range(self.freez + obsv_list.index(obsv) + 1):
                        if obsv_mouths < self.freez:
                            res['profit'] +=  self.coupon_monthly
                        else:
                            res['profit'] +=  (self.coupon_yearly - down_ret*(obsv_mouths - self.freez + 1))/12
                    #res['profit'] = (self.coupon_yearly + 1) ** ((self.freez + obsv_list.index(obsv) + 1) / 12) - 1
                    return res
            else:
                sum_down = down_price *((self.freez + obsv_list.index(obsv)) - KO_decrease_months)
                if self.data['close'][obsv] >= self.CLOSE * (self.KO - sum_down):
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    for obsv_mouths in range(self.freez + obsv_list.index(obsv) + 1):
                        if obsv_mouths < self.freez:
                            res['profit'] +=  self.coupon_monthly
                        else:
                            res['profit'] +=  (self.coupon_yearly - down_ret*(obsv_mouths - self.freez + 1))/12
                    # res['profit'] = (self.coupon_yearly + 1 - down_ret * (self.freez + obsv_list.index(obsv) + 1 - self.freez)) \
                    #                     ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    return res
        # 敲入
        price_min = min(self.data['close'][i:i+252])
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            for obsv_mouths in range(self.freez + obsv_list.index(obsv) + 1):
                if obsv_mouths < self.freez:
                    res['profit'] +=  self.coupon_monthly
                else:
                    res['profit'] +=  (self.coupon_yearly - down_ret*(obsv_mouths - self.freez + 1))/12            
            # res['profit'] = (self.coupon_yearly + 1 - down_ret * (self.freez + obsv_list.index(obsv) + 1 - self.freez)) - 1
            res['out_price'] = self.data['close'][obsv_list[-1]]
            return res
        
    #经典雪球
    def CLA(self, obsv_list:list):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
        }
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                res['date'] = self.data['Date'][obsv]
                res['FLAG'] = 1
                res['out_price'] = self.data['close'][obsv]
                #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
                return res
        # 敲入
        price_min = min(self.data['close'][i:i+252])
         
        
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE * self. KI:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_yearly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            return res
        
        
        
if __name__ == '__main__':
    print("启动模拟时间：", datetime.now())    
    # 计时开始
    tic = time()
    # 初始化
    snowBall = snowBalls(type='CLA', year=1, coupon_yearly=0.175, freez=3, KO = 1.03, KI = 0.75)
    # 运行蒙特卡洛模拟
    snowBall.run()
    # 计时结束 
    tok = time()

    print("运行时间：", round(tok - tic, 2), 's')
