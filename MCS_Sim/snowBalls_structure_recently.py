import numpy as np
from datetime import datetime
import pandas as pd

class Structure:
    
    def __init__(self, ):
        
        self.i = None
        self.data = None
        self.CLOSE = None
        self.KO = None
        self.KI = None
        self.freez = None
        self.coupon_yearly = None
        self.coupon_monthly = None
        self.year = None
        
        
    
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
                res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                return res
        # 敲入
        price_min = np.min(self.data['close'][i:i+252])
         
        
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE * self. KI:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE * self. KI - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_yearly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
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
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res
            else:
                sum_down = down *((self.freez + obsv_list.index(obsv)) - KO_decrease_months)
                if self.data['close'][obsv] >= self.CLOSE * (self.KO - sum_down):
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res
        # 敲入
        price_min = np.min(self.data['close'][i:i+252])
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_yearly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
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
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
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
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res
        # 敲入
        price_min = min(self.data['close'][i:i+252])
        if price_min < self.CLOSE *self.KI:            
            if self.data['close'][obsv_list[-1]] < self.CLOSE:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][obsv_list[-1]] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
            else:
                res['date'] = self.data['Date'][obsv_list[-1]]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
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
            res['lasting'] = self.freez + len(obsv_list)
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
                res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                return res
    # 敲入
        price_min = min(self.data['close'][i : i + self.year * 252])
        
        if i > len(self.data) - 252 * self.year:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
            
        if price_min < self.CLOSE * self.KI:            
            if self.data['close'][last_date] < self.CLOSE:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][last_date] ) / self.CLOSE)
                res['out_price'] = self.data['close'][obsv_list[-1]]
                res['lasting'] = self.freez + len(obsv_list)
                return res
            else:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  0
                res['out_price'] = self.data['close'][last_date]
                res['lasting'] = self.freez + len(obsv_list)
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][last_date]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_monthly * 12
            res['out_price'] = self.data['close'][last_date]
            res['lasting'] = self.freez + len(obsv_list)
            return res
        
    #经典雪球+看涨期权
    def CLA_call(self, obsv_list:list):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
        }
        
        price_min = min(self.data['close'][i : i + self.year * 252])
        
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                # 曾敲入
                if price_min < self.CLOSE * self.KI:   
                    res['date'] = str(self.data['Date'][obsv]) + '*'
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    # 复利
                    #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    # 单利
                    # res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1) + (self.data['close'][obsv] - self.CLOSE * self.KI)/(self.CLOSE * self.KI)
                    # 加期权费
                    res['profit'] = (self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1) + (self.data['close'][obsv] - self.CLOSE * self.KI)/(self.CLOSE * self.KI) - self.call_cost) / (1 + self.call_cost)
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res
                # 不曾敲入
                else:
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res                    
        # 敲入
        if i > len(self.data) - 252 * self.year:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
            
        if price_min < self.CLOSE * self.KI:                    
            if self.data['close'][last_date] < self.CLOSE:
                # 到期日小于行权价，则不行权    
                if self.data['close'][last_date] < self.CLOSE * self.KI:
                    res['date'] = self.data['Date'][last_date]
                    res['FLAG'] = -1
                    res['profit'] =  ((self.data['close'][last_date] - self.CLOSE) / self.CLOSE - self.call_cost) / (1 + self.call_cost)
                    res['out_price'] = self.data['close'][last_date]
                    res['lasting'] = self.freez + len(obsv_list)
                    return res
                else:
                    res['date'] = self.data['Date'][last_date]
                    res['FLAG'] = -1
                    res['profit'] =  ((self.data['close'][last_date] - self.CLOSE) / self.CLOSE + (self.data['close'][last_date] - self.CLOSE * self.KI)/(self.CLOSE * self.KI) - self.call_cost) / (1 + self.call_cost)
                    res['out_price'] = self.data['close'][last_date]
                    res['lasting'] = self.freez + len(obsv_list)
                    return res
            else:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  (0 + (self.data['close'][last_date] - self.CLOSE * self.KI)/(self.CLOSE * self.KI) - self.call_cost) / (1 + self.call_cost)
                res['out_price'] = self.data['close'][last_date]
                res['lasting'] = self.freez + len(obsv_list)
                return res
        # 稳定
        else:
            res['date'] = self.data['Date'][last_date]
            res['FLAG'] = 0
            res['profit'] = self.year * self.coupon_monthly * 12
            res['out_price'] = self.data['close'][last_date]
            res['lasting'] = self.freez + len(obsv_list)
            return res