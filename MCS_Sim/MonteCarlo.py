import pandas as pd
import numpy as np
from numpy import random
import os
from collections import deque
from queue import Queue
from numba.experimental import jitclass
from numba import njit, types, typed
from concurrent.futures import ProcessPoolExecutor
from pprint import pprint


class MCS:
    #     Monte Carlo Simulation
    #     使用多进程加速
    #     使用 Numba - jitclass加速 - 未实现
    #     @author: cxy

    # private:
    def __init__(self, times=int(1e+3), mode='sim', type='ETF', year=2):
        #     存储数据文件信息
        self.START = 978
        # self.FROM  = self.START - 21 * 3
        self.FROM = 0

        self._DataDict = {
            'trade_date': None,
            'open_price': None,
            'close': None,
            'pe': None,
            'close_simulation': None,
            'pe_simulation': None,
            'future_price' : None,
            'abr' : None,
            'vol' : None,  # 60日年化波动率，数据来源：Wind
            'vol_simulation' : None,
            'sim_array' : None,
            'MA' : None,
        }

        #     存储结果信息
        self._ResDict = {
            'time_k': None,
            'set_up': None,
            'cl': None,
            'p_v': None,
            'knock_in': 0,
            'knock_out': 0,
            'knock_stable': 0,
            'MU' : [],
            'Median' : [],
            # 'vol': deque()
        }
        #     设置参数信息
        self._ParamDict = {
            'times': times,  # 模拟次数：1e+5 10万次模拟
            'mode' : mode,
            'type' : type,
            'start': self.START,
            'length': 1,  # 模拟的天数  没用到
            'year': year,  # 雪球结构时长
            'KO_ratio': 1,  # 敲出系数
            'KI_ratio': 0.8,  # 敲入系数
        }
        #     设置计算量
        self._Coefficient = {
            'rf': None,
            'd': None,
            'n': None,
            'dt': None,
            'sigma_simulation': None,
            'eg': [],
            'pe_0': 15,
        }

    # pbulic:
    def loadData(self, ):
        """
        加载数据函数 数据文件需要与该脚本在同一路径下
        数据文件名：Temp.xlsx
        :return: None
        """
        filename = 'M.DCE.xlsx'
        path = os.getcwd() + '\\DATA\\' + filename
        data = pd.read_excel(path, 'Sheet1')

        #         设置 _DataDict
        self._DataDict['trade_date'] = data['Date']

        self._DataDict['close'] = data['close']

        self._DataDict['abr'] = data['anal_basisannualyield']

        self._DataDict['pe'] = data['val_pe_ttmwgt']

        self._DataDict['vol'] = data['volatilityratio']

        self._Coefficient['eg'] = data['eg']

        if 'MA' in data.columns:
            self._DataDict['MA'] = data['MA']


    def setData(self, ):
        """
        数据初始化
        :return: None
        """
        year_ = 1

        self._ParamDict['start'] = self.START

        start = self._ParamDict['start']

        close = self._DataDict['close']

        pe = self._DataDict['pe']

        year = self._ParamDict['year']

        sim_lenth = int(year * 252) + 1

        self._DataDict['close_simulation'] = close[int(start - 252 * year_): start]
        # self._DataDict['close_simulation'] = close[int(start - 1): start]

        self._DataDict['vol_simulation'] = self._DataDict['vol'][int(start - 1): start]

        self._Coefficient['sigma_simulation'] = self._DataDict['vol_simulation'].iloc[-1] / 100

        self._DataDict['sim_array']  = list(close[start : start + sim_lenth])

        self._Coefficient['dt'] = 1 / (252 * year)  # 单位时间

    def mcs(self):
        """
        蒙特卡洛模拟主循环
        :return:
        """
        mode = self._ParamDict['mode']
        year = self._ParamDict['year']
        if mode == 'cal':
            if self._ParamDict['type'] == 'future2':
                args = (0, 0, 0)
                iter_params = [(self.mcs_init(*args, mode=mode), args)]
        elif mode == 'sim':
            pe_504_List = [15, 20, 30]  #
            eg_List = [0.05, 0.07, 0.1]  # 
            sigma_List = [0.1, 0.2, 0.3]  #
            args = [(pe_504, eg, sigma_simulation) for pe_504 in pe_504_List \
                                                 for eg in eg_List \
                                                 for sigma_simulation in sigma_List]
            iter_params = [(self.mcs_init(*arg, mode=mode), arg) for arg in args]
        elif mode == 'backTrade':
            pass

        q = Queue()
        with ProcessPoolExecutor() as pool:
            # for i in range(len(iter_params)):
            #     q.put((args[i], self.mcs_iter(*iter_params[i])))
            for iter_param, pn_mat in zip(iter_params, pool.map(self.mcs_iter, iter_params)):
                q.put((iter_param[1], pn_mat))

        return q

    def mcs_init(self, pe_504_, eg_, sigma_simulation_, mode='cal'):
        # 读取参数
        type = self._ParamDict['type']
        START_PRICE = self._DataDict['close_simulation'].iloc[-1]
        future_price = START_PRICE * (1 - 0.053)
        times = self._ParamDict['times']
        year = self._ParamDict['year']
        dt = self._Coefficient['dt']
        sqrt_dt = np.sqrt(dt)
        pe_0 = self._Coefficient['pe_0']
        price_504 = 0
        _delta = 0

        if mode == 'cal':
            sigma_simulation = self._Coefficient['sigma_simulation']
            eg = self._Coefficient['eg']
        elif mode == 'sim':
            pe_504, eg, sigma_simulation = pe_504_, eg_, sigma_simulation_

            earning_0 = START_PRICE / pe_0
            earning_504 = earning_0 * (1 + eg) ** year
            price_504 = pe_504 * earning_504

        ##################################################
        if type == 'ETF':
            MU = (price_504 / START_PRICE) ** (252 / (252 * year)) - 1
        elif type == 'future1':
            abr = ((START_PRICE - future_price) / START_PRICE ) ** (1 / year)
            pg = (price_504 - START_PRICE) / START_PRICE
            MU = pg - abr + _delta
        elif type == 'future2':
            ##################################################
            # abr = self._DataDict['abr'][self.START-1]
            # eg  = (1 + np.nanmedian(self._Coefficient['eg'][self.FROM: self.START])) ** 252 - 1
            # eg = self._Coefficient['eg'][self.START-1]
            # MU = eg + (-abr/100) + _delta
            ##################################################
            median = np.median(self._DataDict['close_simulation'])
            MU = (1 + ((median - START_PRICE) / START_PRICE)) ** year - 1
            print("meian: ", median)
            self._ResDict['Median'].append(median)
            # MA = self._DataDict['MA'][self.START-1]
            # print("ma: ", MA)

            # MU = (1 + ((MA - START_PRICE) / START_PRICE)) ** year - 1

            # print('eg_nanmedian: ', np.nanmedian(self._Coefficient['eg'][self.FROM: self.START]))
            # print('eg:', eg)
            # print('abr:', (-abr/100))
            print('MU:', MU)
            self._ResDict['MU'].append(MU)
        ##################################################

        iter_params = (times, MU, dt, sigma_simulation, sqrt_dt, START_PRICE, price_504, year)

        return iter_params

    def mcs_iter(self, args):
        # price_start = 1
        times, MU, dt, sigma_simulation, sqrt_dt, price_start, price_504, year = args[0]
        h_sigma_sqr = sigma_simulation ** 2 / 2

        nudt = (MU - h_sigma_sqr) * dt

        volsdt = sigma_simulation * sqrt_dt
        lnS = np.log(price_start)

        days = int(year * 12 * 21) + 1
        randomMarix = np.random.normal(size=(times, days))

        delta_lnSt = nudt + volsdt * randomMarix
        lnSt = lnS + np.cumsum(delta_lnSt, axis=1)
        ST = np.exp(lnSt)

        return ST

    def snowKick(self, q):
        """
        根据模拟结果统计观察期内敲入、敲出、稳定的频数
        :param q: 模拟结果队列
        :return: 返回雪球敲入敲出结果
        """
        # 读取设置参数
        year = self._ParamDict['year']
        KO_ratio = self._ParamDict['KO_ratio']
        KI_ratio = self._ParamDict['KI_ratio']
        price_start = self._DataDict['close_simulation'].iloc[-1]
        s = slice(20, int(year * 252) + 1, 21)
        snowKickDict = {
            #             '日期': [self._DataDict['trade_date'][self._ParamDict['start'] - 1]],
            #             '收盘价': [self._DataDict['close'][self._ParamDict['start'] - 1]],
            '敲出次数': [],
            '敲入次数': [],
            '稳定次数': [],
            'pe_504': [],
            'eg': [],
            'vol': [],  # sigma_simulation
        }

        while True:
            self._ResDict['knock_out'], self._ResDict['knock_in'], self._ResDict['knock_stable'] = 0, 0, 0
            if q.empty():
                break  # 出口
            res = q.get()
            for pn_array in res[1]:
                pn_slice = pn_array[s]
                p_max = max(pn_slice)
                p_min = min(pn_array)

                #         敲入敲出条件
                if p_max >= KO_ratio * price_start:
                    self._ResDict['knock_out'] += 1
                elif p_min <= KI_ratio * price_start and p_max < KO_ratio * price_start:
                    self._ResDict['knock_in'] += 1
                elif p_min > KI_ratio * price_start and p_max < KO_ratio * price_start:
                    self._ResDict['knock_stable'] += 1

            snowKickDict['pe_504'].append(res[0][0])
            snowKickDict['eg'].append(res[0][1])
            snowKickDict['vol'].append(res[0][2])
            snowKickDict['敲出次数'].append(self._ResDict['knock_out'])
            snowKickDict['敲入次数'].append(self._ResDict['knock_in'])
            snowKickDict['稳定次数'].append(self._ResDict['knock_stable'])

        return pd.DataFrame(snowKickDict)

    def sigma_flag(self, MinMax: tuple, lenth: int, pn_array: list):
        if len(pn_array) - lenth - 1 >= 0:
            close = pn_array[len(pn_array) - lenth - 1:]
        else:
            close = np.append(self._DataDict['close_simulation'], pn_array, axis=0)
            close = close[len(close) - lenth - 1:]

        #         区间对数收益率
        Ri = np.array([np.log(close[x + 1] / close[x]) for x in range(len(close) - 1)])
        #         区间对数平均收益率
        R_mean = np.mean(Ri)
        #         波动率
        vol_ = np.sqrt((np.sum(np.power((Ri - R_mean), 2))) / (len(Ri) - 1))
        if vol_ <= max(MinMax) and vol_ >= min(MinMax):
            self._ResDict['vol'].append(vol_)
            return True
        else:
            return False

    def run(self, ):
        """
        蒙特卡洛自动运行
        :return: 返回统计结果
        """

        self.loadData()

        self.setData()

        q = self.mcs()

        snowKick = self.snowKick(q)

        return snowKick


    def backTrader(self):
        start = 252
        lenth = 10000
        res_dict = {
            # 'date' : [],
            # 'close' : [],
            # 'eg' : [],
            # 'abr' : [],
            # 'winRate' : [],
            'realRes' : [],
            # 'MU' : None,
            # 'Median' : None,
        }

        self.loadData()

        i = 0
        FLAG = 0
        sim_array = self._DataDict['sim_array']

        while True:
            if FLAG:
                resDF = pd.DataFrame(res_dict)
                break  # 出口

            try:
                self.START = start + i
                self.FROM  = i

                self.setData()
###################################################
                START_PRICE = self._DataDict['close_simulation'].iloc[-1]
                print('Date: ', self._DataDict['trade_date'].iloc[self.START-1])
                print('START_PRICE: ', START_PRICE)
                # print('index: ', self.START)
                # print('close: ', self._DataDict['close_simulation'].iloc[-1])
                # print('vol: ', self._Coefficient['sigma_simulation'])
                #
                # sim_q = self.mcs()
                #
                # sim_snowKick = self.snowKick(sim_q)
                #
                # res_dict['date'].append(self._DataDict['trade_date'].iloc[self.START - 1])
                # winRate = float(round((sim_snowKick['敲出次数'] + sim_snowKick['稳定次数']) / self._ParamDict['times'], 4))
                # print("胜率：", winRate)
                # res_dict['winRate'].append(winRate)
                # res_dict['close'].append(START_PRICE)
                # res_dict['eg'].append(self._Coefficient['eg'][self.START-1])
                # res_dict['abr'].append(self._DataDict['abr'][self.START-1] / 100)
                #
                # print(sim_snowKick)
######################################################
                real_q = Queue()
                real_q.put(((0,0,0), [self._DataDict['sim_array']]))

                try:
                    real_snowKick = self.snowKick(real_q)
                except:
                    res_dict['realRes'].append(None)
                    resDF = pd.DataFrame(res_dict)
                    break

                if int(real_snowKick['敲出次数']) & 1:
                    realRes = 1
                elif int(real_snowKick['敲入次数']) & 1:
                    realRes = -1
                else:
                    realRes = 0
                res_dict['realRes'].append(realRes)

                # res_dict['MU'] = self._ResDict['MU']
                # res_dict['Median'] = self._ResDict['Median']

                print(real_snowKick)

                print('#'*50)


            # 到达回测序列终点
            except:
                pprint(res_dict)
                resDF = pd.DataFrame(res_dict)
                print('err')
                break   # 出口

            if i == lenth:
                FLAG = 1

            i += 1

        resDF.to_excel('backTrade.xlsx', index=False)

        return None
