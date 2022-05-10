import pandas as pd
import numpy as np
from numpy import random
import os
from collections import deque
from queue import Queue
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor


class MCS:
    #     Monte Carlo Simulation
    #     使用多进程加速
    #     使用 Numba加速 - 未实现
    #     @author: cxy

    # private:
    def __init__(self, times=int(1e+3)):
        #     存储数据文件信息
        self._DataDict = {
            'trade_date': None,
            'open_price': None,
            'close': None,
            'pe': None,
            'close_simulation': None,
            'pe_simulation': None,
        }

        #     存储结果信息
        self._ResDict = {
            'time_k': None,
            'set_up': None,
            'cl': None,
            'p_v': None,
            'pt': [],  # 没用到
            'knock_in': 0,
            'knock_out': 0,
            'knock_reverse': 0,  # 没用到
            'knock_stable': 0,
            'vol': deque()
        }
        #     设置参数信息
        self._ParamDict = {
            'times': times,  # 模拟次数：1e+5 10万次模拟
            'start': 3730,
            'length': 1,  # 模拟的天数  没用到
            'year': 2,  # 雪球结构时长
            'KO_ratio': 1,  # 敲出系数
            'KI_ratio': 0.8,  # 敲入系数
            'sentiment': 18,
            'simulation_type': 2,
            'paramList': [],
        }
        #     设置计算量
        self._Coefficient = {
            'rf': None,
            'd': None,
            'n': None,
            'dt': None,
            'sigma_simulation': 0.15,
            'eg': 0.05,
            'pe_0': 15,
        }

    # pbulic:

    def loadData(self, ):
        """
        加载数据函数 数据文件需要与该脚本在同一路径下
        数据文件名：Temp.xlsx
        :return: None
        """
        filename = 'Temp.xlsx'
        path = os.getcwd() + '\\' + filename
        data = pd.read_excel(path)

        #         设置 _DataDict
        self._DataDict['trade_date'] = data['日期']
        self._DataDict['close'] = data['收盘价(元)']
        self._DataDict['open_price'] = data['开盘价(元)']
        self._DataDict['pe'] = data['市盈率']

    def setData(self, ):
        """
        数据初始化
        :return: None
        """
        start = self._ParamDict['start']
        trade_date = self._DataDict['trade_date']
        close = self._DataDict['close']
        pe = self._DataDict['pe']
        year = self._ParamDict['year']

        trade_date_simulation = trade_date[start - 252 * year - 1: start]

        close_simulation = close[start - 252 * year - 1: start]
        self._DataDict['close_simulation'] = close_simulation

        pe_simulation = pe[start - 252 * year - 1: start]
        self._DataDict['pe_simulation'] = pe_simulation

        self._Coefficient['n'] = len(trade_date_simulation)  # 历史价格时间长度

        self._Coefficient['dt'] = 1 / (252 * year)  # 单位时间

    def mcs(self, mode='cal'):
        """
        蒙特卡洛模拟主循环
        :return:
        """
        if mode == 'cal':
            args = (0, 0, 0)
            iter_params = [self.mcs_init(*args, mode=mode)]
        elif mode == 'sim':
            pe_504_List = [15, 20, 30]  #
            eg_List = [0.05, 0.07, 0.1]  # 
            sigma_List = [0.1, 0.2, 0.3]  #
            args = [(pe_504, eg, sigma_simulation) for pe_504 in pe_504_List \
                                                 for eg in eg_List \
                                                 for sigma_simulation in sigma_List]
            iter_params = [(self.mcs_init(*arg, mode=mode), arg) for arg in args]
        else:
            pass

        q = Queue()
        with ProcessPoolExecutor() as pool:
            # for i in range(len(iter_params)):
            #     q.put((args[i], self.mcs_iter(*iter_params[i])))
            for iter_param, pn_mat in zip(iter_params, pool.map(self.mcs_iter, iter_params)):
                q.put((iter_param[1], pn_mat))

        return q

    def mcs_init(self, pe_504_, eg_, sigma_simulation_, mode='cal'):
        # 读取设置参数

        STRAT_PRICE = self._DataDict['close_simulation'].iloc[-1]
        times = self._ParamDict['times']
        year = self._ParamDict['year']
        # sentiment = self._ParamDict['sentiment']
        dt = self._Coefficient['dt']
        sqrt_dt = np.sqrt(dt)
        # pe_simulation = self._DataDict['pe_simulation']
        pe_0 = self._Coefficient['pe_0']

        if mode == 'cal':
            sigma_simulation = self._Coefficient['sigma_simulation']
            eg = self._Coefficient['eg']
        elif mode == 'sim':
            pe_504, eg, sigma_simulation = pe_504_, eg_, sigma_simulation_

        #         初始化random矩阵增加cache命中率
        #         random_mat = np.array([np.random.normal(0, 1, year * 252) for _ in range(times)])

        # if sentiment > 3:
        #     pe_504 = sentiment
        # elif sentiment > 2:
        #     pe_504 = np.quantile(pe_simulation, 0.05, axis=0)
        # elif sentiment > 1:
        #     pe_504 = np.quantile(pe_simulation, 0.1, axis=0)
        # else:
        #     pe_504 = np.quantile(pe_simulation, 0.25, axis=0)

        earning_0 = STRAT_PRICE / pe_0
        earning_504 = earning_0 * (1 + eg) ** year
        price_504 = pe_504 * earning_504  # 一个计算中间值
        # MU = (price_504 / STRAT_PRICE) ** (1 / (252 * year))
        ##################################################
        MU = (price_504 / STRAT_PRICE) ** (1 / (252 * year)) - 1
        ##################################################
        iter_params = (times, MU, dt, sigma_simulation, sqrt_dt, STRAT_PRICE, price_504, year)

        return iter_params

    def mcs_iter(self, *args):
        times, MU, dt, sigma_simulation, sqrt_dt, price_start, price_504, year = args[0][0]
        # price_start = 1
        pn_mat = deque()
        lenth = range(1, year * 12 * 21)
        h_sigma_sqr = sigma_simulation**2 / 2
        sqrt252 = np.sqrt(252)
        days = year * 252
        while True:
            price_pre = price_start
            mu_simulation = MU
            if times == 0:
                break
            pn_array = []
            #             每次模拟
            for day in lenth:
                # FLAG 为True表示一次合格的模拟
                FLAG = 0
                while True:
                    pn = np.exp((mu_simulation - h_sigma_sqr) * dt + sigma_simulation * sqrt_dt * np.random.normal(0, 1)) * price_pre
                    pn_array_temp = pn_array + [pn]
                    ##################### 设置合格条件 ##################
                    if self.sigma_flag(MinMax=(0, 10000), lenth=20, pn_array=pn_array_temp):
                    # if 1==1:
                        FLAG = 1
                    ####################################################
                    if FLAG:
                        pn_array.append(pn)
                        del pn_array_temp
                        # 出口
                        break
                    del pn_array_temp
                ###################### 设置滚动参数 #####################
                mu_simulation = (price_504 / pn) ** (252 / (days - day)) - 1

                price_pre = pn
                ########################################################

            pn_mat.append(deepcopy(pn_array))
            times -= 1

        return pn_mat

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
        price_start = self._DataDict['close'].iloc[-1]
        s = slice(20, year * 252, 21)
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
                break
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

    def run(self, mode: str):
        """
        蒙特卡洛自动运行
        :return: 返回统计结果
        """
        if mode not in ('cal', 'sim'):
            print("Unexpected mode")
            print("mode: 'cal' or 'sim'")

        self.loadData()

        self.setData()

        q = self.mcs(mode=mode)

        snowKick = self.snowKick(q)

        return snowKick