import pandas as pd
import numpy as np
from pprint import pprint
import os
from datetime import datetime
from time import time
from snowBalls_structure import Structure
import warnings
warnings.filterwarnings("ignore")


class snowBalls:

    def __init__(self,):

        self.data = None
        self.CLOSE = None
        self.i = None
        self.showon = 0
        self.structure = None
        self.type = 'CLA_call_boost'       # 1.经典雪球：CLA; 2.经典雪球+买入期权：CLA_call; 3.OTM: OTM; 4.救生舱：LBT; 5.CLA_boost:经典雪球+指数增强产品
        self.freez = 2          # 封闭期
        # self.year = 2           # 雪球时长
        self.duration = 24           # 雪球时长
        self.max_duration = self.duration
        self.KO = 1.0           # 敲出系数
        self.KO_LBT = 0.85      # 救生艇敲出系数
        self.KO_fix = self.KO           # 敲出系数
        self.KI = 0.80          # 敲入系数
        self.PE_style = 0       # PE计算方式：0为静态，1为滚动
        self.PE_static_thsh = 100       # 静态PE阈值，数值为PE绝对数值
        self.PE_trailing_thsh = 25        # 滚动PE阈值，数值为历史周期分位数
        self.PE_trailing_lenth = 252       # 滚动PE历史周期长度，数值单位：日
        self.coupon_yearly = 0.0635    #年利率
        self.coupon_yearly_LBT = 0.18    #年利率_触发救生舱条件后
        self.obsv_date_LBT = 6          #重置观察期
        self.coupon_monthly = self.coupon_yearly / 12   # 月票息
        self.N = 3 /100             # 大于年化 N% 概率阈值
        self.call_cost = 0.05     #期权费
        self.begindate = '2004-12-31'
        self.enddate = '2022-08-03'
        # 针对指数增强
        self.is_fix = 0 # 产品是否固定 0:不固定
        self.length_boost = 24 # 指增时长
        self.a_boost = 0.05 # 指增超额收益
        # self.recent_year = 0        # 1:回测最近一年；0：回测一年前
        self.freez_ = self.freez
        

    def loadData(self, filename=''):
        
        begindate = datetime.strptime(self.begindate, "%Y-%m-%d")
        enddate = datetime.strptime(self.enddate, "%Y-%m-%d")
        path = os.getcwd() + '\\' + '000852.xlsx' + filename
        self.data = pd.read_excel(path, 'Sheet1')
        self.data = self.data[(snowBall.data['Date'] >= begindate) & (snowBall.data['Date'] <= enddate)]
        self.data = self.data.reset_index()
        self.init_snowBalls()
        
        
    def init_snowBalls(self):
        
        self.structure = Structure()
        self.structure.data = self.data
        self.structure.KO = self.KO
        self.structure.KI = self.KI
        self.structure.freez = self.freez
        self.structure.coupon_yearly = self.coupon_yearly
        self.structure.coupon_monthly = self.coupon_monthly
        # self.structure.year = self.year
        self.structure.call_cost = self.call_cost
        self.structure.KO_LBT = self.KO_LBT
        self.structure.KO_fix = self.KO_fix
        self.structure.coupon_yearly_LBT = self.coupon_yearly_LBT 
        self.structure.obsv_date_LBT = self.obsv_date_LBT
        self.structure.duration = self.duration 
        self.structure.is_fix =  self.is_fix # 产品是否固定
        self.structure.length_boost = self.length_boost # 指增长度
        self.structure.a_boost = self.a_boost # 指增超额收益


    def obsv_date(self, i):

        pre, j = i, i + 1  # 双指针
        # if i > len(self.data) - 252 * self.year:
        #     print('over')
        #     return None
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

                if count >= self.freez and count <= self.duration - 1 and month_now != pre_month:

                    obsv_list.append(pre)
                    count += 1
                    pre_month = self.data['Date'][j].month
                if count > self.duration - 1:
                    FLAG = 1
                    # 出口
                    break
            elif day_now < day_pre and day_pre < base_day:
                if count < self.freez and month_now != pre_month:
                    count += 1
                    pre_month = self.data['Date'][pre].month
                if count >= self.freez and count <= self.duration - 1 and month_now != pre_month:
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


    def backTrader(self, recent_year=0):

        res_dict = {
            'Date': [],
            'close' : [],
            'PE' : [],
            'realRes': [],
            'knockOUT_date' : [],
            'knockIN_date' : [],
            'out_price' : [],
            'lasting' : [],
            'profits' : [],
            'annually' : [],
            'ever_in' : [],
            "+call" : [],
        }
        self.recent_year = recent_year
        if self.is_fix == 0:
            self.max_duration = self.duration + self.length_boost
        for i in range(len(self.data)):            
            self.i = i
            self.structure.i = self.i
            thsh = 0
            if self.recent_year:
                if i >= len(self.data) - 21 * self.freez_:
                    self.freez = 0
                    self.structure.freez = 0
                    if self.data['pe_ttm'][i] >= self.get_PE_thsh(i):
                        thsh = 1  # PE高于阈值
                    self.CLOSE = self.data['close'][i]
                    self.structure.CLOSE = self.CLOSE
                    obsv_list = self.obsv_date(i)
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
                            res_dict['knockOUT_date'].append(None)
                            res_dict['knockIN_date'].append(None)
                            res_dict['ever_in'].append(None)
                            res_dict['+call'].append(None)
                    else:
                        res_dict['Date'].append(self.data['Date'][i])
                        res_dict['close'].append(self.CLOSE)
                        res_dict['PE'].append(self.data['pe_ttm'][i])
                        res_dict['realRes'].append(res['FLAG'])
                        res_dict['out_price'].append(res['out_price'])
                        res_dict['lasting'].append(res['lasting'])
                        res_dict['profits'].append(res['profit'])
                        annually = (1 + res['profit']) ** (252 / (self.data[self.data['Date'] == res['date']].index[0] - i)) - 1
                        res_dict['annually'].append(annually)
                        res_dict['ever_in'].append(res['ever_in'])
                        try:
                            res_dict['+call'].append(res['+call'])
                        except:
                            res_dict['+call'].append("")
                        if res['FLAG'] == 1:
                            res_dict['knockOUT_date'].append(res['date'])
                            res_dict['knockIN_date'].append(np.nan)
                        elif res['FLAG'] == 0:
                            res_dict['knockOUT_date'].append(np.nan)
                            res_dict['knockIN_date'].append(np.nan)
                        elif res['FLAG'] == -1:
                            res_dict['knockOUT_date'].append(np.nan)
                            res_dict['knockIN_date'].append(res['date'])

            else:
                if self.data['pe_ttm'][i] >= self.get_PE_thsh(i):
                    thsh = 1  # PE高于阈值
                self.CLOSE = self.data['close'][i]
                self.structure.CLOSE = self.CLOSE
                obsv_list = self.obsv_date(i)
                # 如果在雪球期限之内则返回None
                if i >= len(self.data) - 21 * self.freez_:
                    obsv_list = None
    
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
                            res_dict['knockOUT_date'].append(None)
                            res_dict['knockIN_date'].append(None)
                            res_dict['ever_in'].append(None)
                            res_dict['+call'].append(None)
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
                        res_dict['ever_in'].append(res['ever_in'])
                        try:
                            res_dict['+call'].append(res['+call'])
                        except:
                            res_dict['+call'].append("")
                        if res['FLAG'] == 1:
                            res_dict['knockOUT_date'].append(res['date'])
                            res_dict['knockIN_date'].append(np.nan)
                        elif res['FLAG'] == 0:
                            res_dict['knockOUT_date'].append(np.nan)
                            res_dict['knockIN_date'].append(np.nan)
                        elif res['FLAG'] == -1:
                            res_dict['knockOUT_date'].append(np.nan)
                            res_dict['knockIN_date'].append(res['date'])
                    # print(i,res_dict['Date'][-1],res['date'],res['lasting'],self.freez)
    
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
            '分类' : ['敲入', '', '', '敲出', '', '稳定', '平均敲出月份数', '总计', '亏损', '', '', '', '', '', '盈利', '', '', '', '', '', '', '年化收益>{}%的概率'.format(self.N*100), '绝对收益>{}%的概率'.format(self.N*100)],
            '情形' : ['盈利', '亏损', '不亏不赚', '盈利_未敲入', '盈利_曾敲入', '盈利', '', '', '亏损最大值', '亏损1/4', '亏损2/4', '亏损3/4', '亏损最小值', '平均数', '盈利最大值', '盈利1/4', '盈利2/4', '盈利3/4', '盈利最小值', '平均数', '', '', ''],
            '次数' : ['' for _ in range(23)],
            '平均收益/亏损' : ['' for _ in range(23)],
            '不加票息' : ['' for _ in range(23)],
        }
        profits_kickOUT = [df['profits'][i] for i in range(len(df))
                                            if df['realRes'][i] == 1]
        profits_kickOUT_everin = [df['profits'][i] for i in range(len(df))
                                            if ((df['realRes'][i] == 1) & (df['ever_in'][i] == 1))]
        profits_kickOUT_neverin = [df['profits'][i] for i in range(len(df))
                                            if ((df['realRes'][i] == 1) & (df['ever_in'][i] != 1))]
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
        stat_dict['次数'][1] = (sum(profit < 0 for profit in profits_kickIN))
        stat_dict['次数'][2] = (sum(profit == 0 for profit in profits_kickIN))
        stat_dict['次数'][3] = (sum(profit > 0 for profit in profits_kickOUT_neverin))
        stat_dict['次数'][4] = (sum(profit > 0 for profit in profits_kickOUT_everin))
        stat_dict['次数'][5] = (sum(profit > 0 for profit in profits_stable))
        stat_dict['次数'][7] = sum(stat_dict['次数'][:6])
        try:
            stat_dict['次数'][8] = (np.nanmin(LOST))
            stat_dict['次数'][12] = (max(LOST))
        except:
            stat_dict['次数'][8] = np.nan
            stat_dict['次数'][12] = np.nan
        stat_dict['次数'][9] = (np.nanpercentile(LOST, 75))
        stat_dict['次数'][10] = (np.nanpercentile(LOST, 50))
        stat_dict['次数'][11] = (np.nanpercentile(LOST, 25))
        stat_dict['次数'][13] = (np.nanmean(LOST))
        stat_dict['次数'][14] = (max(WIN))
        stat_dict['次数'][15] = (np.nanpercentile(WIN, 25))
        stat_dict['次数'][16] = (np.nanpercentile(WIN, 50))
        stat_dict['次数'][17] = (np.nanpercentile(WIN, 75))
        stat_dict['次数'][18] = (min(WIN))
        stat_dict['次数'][19] = (np.nanmean(WIN))

        stat_dict['次数'][21] = (annually_profit_len / len(df))
        stat_dict['次数'][22] = (abs_profit_len / len(df))

        stat_dict['平均收益/亏损'][0] = np.nanmean([profit for profit in profits_kickIN if profit > 0])
        stat_dict['平均收益/亏损'][1] = np.nanmean([profit for profit in profits_kickIN if profit < 0])
        stat_dict['平均收益/亏损'][3] = np.nanmean([profit for profit in profits_kickOUT_neverin])
        stat_dict['平均收益/亏损'][4] = np.nanmean([profit for profit in profits_kickOUT_everin])
        stat_dict['平均收益/亏损'][5] = np.nanmean([profit for profit in profits_stable])


        # 平均敲出月份数
        stat_dict['情形'][6] = np.nanmean([df['lasting'][i] for i in range(len(df)) if df['realRes'][i] == 1])
        
        
        if self.type == 'FCN':
            coupon = self.coupon_monthly
        else:
            coupon = 0
        stat_dict['不加票息'][0] = np.nanmean([profit - coupon * self.duration for profit in profits_kickIN if profit > 0])
        stat_dict['不加票息'][1] = np.nanmean([profit - coupon * self.duration for profit in profits_kickIN if profit <= 0])

        return pd.DataFrame(stat_dict)




    def snowKick(self, obsv_list:list,):
        type = self.type
        ####################
        # 字典模拟 switch
        ####################
        if type == 'FCN':
            res = self.structure.FCN(obsv_list)
        if type == 'OTM':
            res = self.structure.OTM(obsv_list)
        if type == 'SSS':
            res = self.structure.SSS(obsv_list)
        if type == 'DSS':
            res = self.structure.DSS(obsv_list)
        if type == 'CLA':
            res = self.structure.CLA(obsv_list)
        if type == 'CLA_call':
            res = self.structure.CLA_call(obsv_list)
        if type == 'CLA_boost':
            res = self.structure.CLA_boost(obsv_list)
        if type == 'LBT':
            res = self.structure.LBT(obsv_list)
        if type == 'CLA_call_boost':
            res = self.structure.CLA_call_boost(obsv_list)
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
        if self.PE_style == 1:
            PE_style = '滚动计算'
        else:
            PE_style = '静态设置'
        parameters = ['雪球类型：{}'.format(self.type),'封闭期：{}个月'.format(self.freez+1),'雪球期限：{}个月'.format(self.duration),\
                      '年化票息：{}'.format(self.coupon_yearly),'年化票息_触发救生舱条件后：{}'.format(self.coupon_yearly_LBT),\
                          '重置观察期：{}'.format(self.obsv_date_LBT),'敲出系数：{}'.format(self.KO),'救生艇敲出系数：{}'.format(self.KO_LBT),\
                              '敲入系数：{}'.format(self.KI),'PE计算方式：{}'.format(PE_style),'期权费：{}'.format(self.call_cost),\
                              '开始日期：{}'.format(self.begindate),'结束日期：{}'.format(self.enddate), \
                                  '','','','','','','','','','']
        parameters = pd.DataFrame(parameters,columns=['参数'])
        
        # df = self.backTrader()
        # df_stat = self.stat(df)
        # df1 = self.backTrader(1)
        # df_stat1 = self.stat(df1)
        # df_stat_merge = pd.concat([df_stat,df_stat1],axis=1)
        # df_stat_merge = pd.concat([df_stat_merge,parameters],axis=1)
        # # print('lenth:', len(df))
        # # pprint(df.head())
        # # df.to_excel('res.xlsx', 'Sheet2', index=False)
    
        # self.showon = 1
        # df_full = self.backTrader()
        # df_full1 = self.backTrader(1)
        # df_full_merge = pd.concat([df_full,df_full1],axis=0)
        
        df = self.backTrader()
        df_stat = self.stat(df)
        self.showon = 1
        df_full = self.backTrader()
        self.showon = 0

        print("done1")
        df1 = self.backTrader(1)
        df_stat1 = self.stat(df1)
        self.showon = 1
        df_full1 = self.backTrader(1)
        print("done2")

        df_stat_merge = pd.concat([df_stat, df_stat1], axis=1)
        df_stat_merge = pd.concat([df_stat_merge, parameters], axis=1)
        df_full_merge = pd.concat([df_full, df_full1], axis=0)
        
        
        df_full_merge.to_excel(writer, 'Sheet1', index=False)
        df_stat_merge.to_excel(writer, 'Sheet2', index=False)
        # 保存
        writer.save()

        return None
    


        
if __name__ == '__main__':

    print("启动回测时间：", datetime.now())
    # 计时开始
    tic = time()
    # 初始化
    snowBall = snowBalls()
    # 运行回测

    snowBall.run()
    # 计时结束 
    tok = time()

    print("运行时间：", round(tok - tic, 2), 's')

