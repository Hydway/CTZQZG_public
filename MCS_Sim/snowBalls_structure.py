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
        self.call_cost = None
        
    
    def FCN(self, obsv_list: list):

        res = {
            'date': None,  # 出场日期
            'FLAG': None,
            'profit': None,
            'out_price': None,
            'lasting': None,
            'obsv': None,
            'ever_in': None,
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
            res['profit'] = self.duration * self.coupon_monthly - \
                            ((self.CLOSE * self.KI - self.data['close'][obsv_list[-1]]) / self.CLOSE)

            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
            res['obsv'] = obsv_list[-1]
            return res
        # 稳定
        else:
            res['date'] = self.data['Date'][obsv_list[-1]]
            res['FLAG'] = 0
            res['profit'] = self.duration * self.coupon_monthly
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
            'ever_in': None,
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
        price_min = min(self.data['close'][i : i + self.duration * 21])
        
        if i > len(self.data) - 21 * self.duration:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
            
        if price_min < self.CLOSE * self.KI:            
            if self.data['close'][last_date] < (self.CLOSE * self.KI):
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE * self.KI - self.data['close'][last_date] ) / self.CLOSE)
                res['out_price'] = self.data['close'][last_date]
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
            if i > len(self.data) - 21 * self.duration:
                res['profit'] = self.coupon_yearly * (len(self.data) - i) / 252
            else: 
                res['profit'] = self.duration * self.coupon_monthly
            res['out_price'] = self.data['close'][last_date]
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
            'ever_in': None,
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
            res['profit'] = self.duration * self.coupon_monthly
            res['out_price'] = self.data['close'][obsv_list[-1]]
            res['lasting'] = self.freez + len(obsv_list)
            return res


    # 敲出价、收益率递减（双降）
    def DSS(self, obsv_list:list, KO_decrease_months=6, down_price=0.005, down_ret=0.01):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
            'ever_in': None,
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
            'ever_in': None,
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
        price_min = min(self.data['close'][i : i + self.duration * 21])
        
        if i > len(self.data) - 21 * self.duration:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
            
        if price_min < self.CLOSE * self.KI:            
            if self.data['close'][last_date] < self.CLOSE:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][last_date] ) / self.CLOSE)
                res['out_price'] = self.data['close'][last_date]
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
            if i > len(self.data) - 21 * self.duration:
                res['profit'] = self.coupon_yearly * (len(self.data) - i) / 252
            else: 
                res['profit'] = self.duration * self.coupon_monthly 
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
            'ever_in': None,
        }
        
        
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                price_min_out = min(self.data['close'][i : obsv+1])
                # 曾敲入
                if price_min_out < self.CLOSE * self.KI:   
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    # 复利
                    #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    # 单利
                    # res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1) + (self.data['close'][obsv] - self.CLOSE * self.KI)/(self.CLOSE * self.KI)
                    # 加期权费
                    res['profit'] = (self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1) + (self.data['close'][obsv] - self.CLOSE * self.KI)/(self.CLOSE * self.KI) - self.call_cost) / (1 + self.call_cost)
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    res['ever_in'] = 1
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
        if i > len(self.data) - 21 * self.duration:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
        
        price_min = min(self.data['close'][i : i + self.duration * 21])
        
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
            if i > len(self.data) - 21 * self.duration:
                res['profit'] = self.coupon_yearly * (len(self.data) - i) / 252
            else: 
                res['profit'] = self.duration * self.coupon_monthly 
            res['out_price'] = self.data['close'][last_date]
            res['lasting'] = self.freez + len(obsv_list)
            return res
    
    #经典雪球+指数增强
    def CLA_boost(self, obsv_list:list):
        # is_fix 产品是否固定
        # length_boost 指增长度
        # a_boost 指增超额收益
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
            'ever_in': None,
        }
        
        
        # 敲出
        for obsv in obsv_list:
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                price_min_out = min(self.data['close'][i : obsv+1])
                # 不曾敲入
                if price_min_out > self.CLOSE * self.KI:   
                    res['date'] = self.data['Date'][obsv]
                    res['FLAG'] = 1
                    res['out_price'] = self.data['close'][obsv]
                    #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                    res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
                    res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                    return res  
         
        # 敲入
        if i > len(self.data) - 21 * self.duration:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
        
        price_min = min(self.data['close'][i : i + self.duration * 21])
        
        if price_min < self.CLOSE * self.KI:   
            if self.is_fix == 0:
                data_thisday = self.data.iloc[i : i + self.duration * 21,:]
                date_in = data_thisday[data_thisday['close'] <= self.CLOSE * self.KI].index[0]
                last_date = date_in + self.length_boost * 21
                if last_date > self.data.shape[0]-1:
                    last_date = self.data.shape[0]-1
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  (self.KI - 1) + self.KI * ((self.data['close'][last_date] - self.CLOSE * self.KI) / (self.CLOSE * self.KI) + self.a_boost)
                res['out_price'] = self.data['close'][last_date]
                res['lasting'] = (last_date - i + 1)/21
            else:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  (self.KI - 1) + self.KI * ((self.data['close'][last_date] - self.CLOSE * self.KI) / (self.CLOSE * self.KI) + self.a_boost)
                res['out_price'] = self.data['close'][last_date]
                res['lasting'] = self.freez + len(obsv_list)
            return res
        # 稳定
        else:
            res['date'] = self.data['Date'][last_date]
            res['FLAG'] = 0
            if i > len(self.data) - 21 * self.duration:
                res['profit'] = self.coupon_yearly * (len(self.data) - i) / 252
            else: 
                res['profit'] = self.duration * self.coupon_monthly 
            res['out_price'] = self.data['close'][last_date]
            res['lasting'] = self.freez + len(obsv_list)
            return res
    
    
    
    
    # 救生艇雪球lifeboat
    def LBT(self, obsv_list:list):
        i = self.i
        res = {
            'date' : None,
            'FLAG' : None,
            'profit': None,
            'out_price' : None,
            'ever_in': None,
        }
        # 敲出
        self.KO = self.KO_fix
        date_index_LBT = None 
        for obsv in obsv_list:
            if (self.duration - (self.freez + obsv_list.index(obsv) + 1) <= self.obsv_date_LBT) \
                & (self.duration - (self.freez + obsv_list.index(obsv) + 1) > 0) & (date_index_LBT == None): 
                # 找到触发救生舱的日期   
                for date_index_mouths in range(obsv + 1,obsv_list[obsv_list.index(obsv)+1] + 1):
                    if self.data['close'][date_index_mouths] <= self.CLOSE * self.KI:
                       self.KO = self.KO_LBT
                       date_index_LBT = date_index_mouths
                       break
                
            if self.data['close'][obsv] >= self.CLOSE * self.KO:
                res['date'] = self.data['Date'][obsv]
                res['FLAG'] = 1
                res['out_price'] = self.data['close'][obsv]
                #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
                #res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
                # 计算收益
                if not date_index_LBT:                    
                    days_sum = int((self.data['Date'][obsv] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1 
                    res['profit'] = self.coupon_yearly * days_sum / 365
                else:
                    days_sum = int((self.data['Date'][obsv] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1 
                    res['profit'] = self.coupon_yearly_LBT * days_sum / 365
                res['lasting'] = self.freez + obsv_list.index(obsv) + 1
                # print(i,res['date'],res['lasting'],self.freez,obsv_list.index(obsv))
                return res
            
        # 敲入        
        if i > len(self.data) - 21 * self.duration:
            last_date = self.data.shape[0]-1
        else:
            last_date = obsv_list[-1]
            
        price_min = min(self.data['close'][i : i + self.duration * 21])
        
        if price_min < self.CLOSE * self.KI:            
            if self.data['close'][last_date] < self.CLOSE:
                res['date'] = self.data['Date'][last_date]
                res['FLAG'] = -1
                res['profit'] =  - ((self.CLOSE - self.data['close'][last_date] ) / self.CLOSE)
                res['out_price'] = self.data['close'][last_date]
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
            if i > len(self.data) - 21 * self.duration:
                days_sum = int((self.data['Date'][last_date] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1 
                res['profit'] = self.coupon_yearly * days_sum / 365
            else: 
                res['profit'] = self.duration * self.coupon_monthly
            res['out_price'] = self.data['close'][last_date]
            res['lasting'] = self.freez + len(obsv_list)
            return res
    
    # # 救生艇雪球lifeboat
    # def LBT_diff(self, obsv_list:list):
    #     i = self.i
    #     res = {
    #         'date' : None,
    #         'FLAG' : None,
    #         'profit': None,
    #         'out_price' : None,
    #     }
    #     # 敲出
    #     self.KO = self.KO_fix
    #     date_index_LBT = None 
    #     for obsv in obsv_list:
    #         if (self.duration - (self.freez + obsv_list.index(obsv) + 1) <= self.obsv_date_LBT) \
    #             & (self.duration - (self.freez + obsv_list.index(obsv) + 1) > 0) & (date_index_LBT == None): 
    #             # 找到触发救生舱的日期   
    #             for date_index_mouths in range(obsv + 1,obsv_list[obsv_list.index(obsv)+1] + 1):
    #                 if self.data['close'][date_index_mouths] <= self.CLOSE * self.KI:
    #                    self.KO = self.KO_LBT
    #                    date_index_LBT = date_index_mouths
    #                    break
                
    #         if self.data['close'][obsv] >= self.CLOSE * self.KO:
    #             res['date'] = self.data['Date'][obsv]
    #             res['FLAG'] = 1
    #             res['out_price'] = self.data['close'][obsv]
    #             #res['profit'] = (self.coupon_yearly+1) ** ((self.freez + obsv_list.index(obsv) + 1)/12) - 1
    #             #res['profit'] = self.coupon_monthly * (self.freez + obsv_list.index(obsv) + 1)
    #             # 计算收益
    #             if not date_index_LBT:                    
    #                 days_sum = int((self.data['Date'][obsv] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1 
    #                 res['profit'] = self.coupon_yearly * days_sum / 365
    #             else:
    #                 days_sum_before = int((self.data['Date'][date_index_LBT] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1
    #                 days_sum_after = int((self.data['Date'][obsv] - self.data['Date'][date_index_LBT]) / np.timedelta64(1, 'D'))
    #                 profit_before = self.coupon_yearly * days_sum_before / 365
    #                 profit_after = self.coupon_yearly_LBT * days_sum_after / 365
    #                 res['profit'] = profit_before + profit_after
    #             res['lasting'] = self.freez + obsv_list.index(obsv) + 1
    #             # print(i,res['date'],res['lasting'],self.freez,obsv_list.index(obsv))
    #             return res
            
    #     # 敲入        
    #     if i > len(self.data) - 21 * self.duration:
    #         last_date = self.data.shape[0]-1
    #     else:
    #         last_date = obsv_list[-1]
            
    #     price_min = min(self.data['close'][i : i + self.duration * 21])
        
    #     if price_min < self.CLOSE * self.KI:            
    #         if self.data['close'][last_date] < self.CLOSE:
    #             res['date'] = self.data['Date'][last_date]
    #             res['FLAG'] = -1
    #             res['profit'] =  - ((self.CLOSE - self.data['close'][last_date] ) / self.CLOSE)
    #             res['out_price'] = self.data['close'][last_date]
    #             res['lasting'] = self.freez + len(obsv_list)
    #             return res
    #         else:
    #             res['date'] = self.data['Date'][last_date]
    #             res['FLAG'] = -1
    #             res['profit'] =  0
    #             res['out_price'] = self.data['close'][last_date]
    #             res['lasting'] = self.freez + len(obsv_list)
    #             return res
    #     # 稳定
    #     else:
    #         res['date'] = self.data['Date'][last_date]
    #         res['FLAG'] = 0
    #         if i > len(self.data) - 21 * self.duration:
    #             days_sum = int((self.data['Date'][last_date] - self.data['Date'][self.i]) / np.timedelta64(1, 'D')) + 1 
    #             res['profit'] = self.coupon_yearly * days_sum / 365
    #         else: 
    #             res['profit'] = self.duration * self.coupon_monthly
    #         res['out_price'] = self.data['close'][last_date]
    #         res['lasting'] = self.freez + len(obsv_list)
    #         return res
    
