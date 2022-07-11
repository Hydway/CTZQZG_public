import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.interpolate import interp1d  #插值用
from scipy.misc import derivative   #求导
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from datetime import datetime
import traceback
from functools import wraps
import seaborn as sns
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(context='notebook', rc=rc)
import matplotlib.pyplot as plt
from pprint import pprint

class lgb:

      def __init__(self):

            self.X = None
            self.Y = None
            self.train_len = 1300
            self.y_pred_full = [0 for _ in range(self.train_len + 252)]
            self.data = None


      def loadData(self, filename=None):
            """

            :param filename:
            :return:
            """
            filename = 'VIX.xlsx'
            path = os.getcwd() + '\\DATA\\VIX\\' + filename
            self.data = pd.read_excel(path, 'Sheet1')

      def cal_dydx(self, y:list, n:int):
            """

            :param y:
            :param n:
            :return:
            """
            insert_num = 5
            t = np.array([int(t_) for t_ in range(len(y))])
            f = interp1d(t, y, kind="cubic")
            xx = np.linspace(t.min(), t.max(), len(y) * insert_num - insert_num + 1)
            fnew = f(xx)
            dy = []
            for x in xx:
                  try:
                        dydx = derivative(f, x + 1e-3, dx=1e-4, n=n, order=5)
                        dy.append(dydx)
                  except:
                        dy.append(dydx)
            s = slice(0, len(dy), insert_num)
            dy = dy[s]
            return dy

      def Symmetry(self, factors):
            """

            :param factors:
            :return:
            """
            col_name = factors.columns
            D, U = np.linalg.eig(np.dot(factors.T, factors))
            S = np.dot(U, np.diag(D ** (-0.5)))

            Fhat = np.dot(factors, S)
            Fhat = np.dot(Fhat, U.T)
            Fhat = pd.DataFrame(Fhat, columns=col_name, index=factors.index)

            return Fhat

      def d_vol(self, df:pd.DataFrame):
            """

            :param df:
            :return:
            """
            df['d1_v250'] = self.cal_dydx(df['volatilityratio250'].tolist(), 1)
            df['d1_v60'] = self.cal_dydx(df['volatilityratio60'].tolist(), 1)
            df['d1_v20'] = self.cal_dydx(df['volatilityratio20'].tolist(), 1)
            df['d2_v250'] = self.cal_dydx(df['volatilityratio250'].tolist(), 2)
            df['d2_v60'] = self.cal_dydx(df['volatilityratio60'].tolist(), 2)
            df['d2_v20'] = self.cal_dydx(df['volatilityratio20'].tolist(), 2)

            return df

      def dataPre(self, df:pd.DataFrame):
            """

            :param df:
            :return:
            """
            df = self.d_vol(df)
            scaler = MinMaxScaler()
            scaler = scaler.fit(df)
            scalered_df = pd.DataFrame(scaler.transform(df), columns=df.columns)

            self.X = df.drop(['realRes'], axis=1)
            self.Y = scalered_df['realRes']


      def init(self, i:int):
            """

            :param i:
            :return:
            """
            train_start = i
            train_lenth = 1300
            gap = 252
            pred_start = train_start + train_lenth + gap
            pred_lenth = 1

            X_ = self.X[train_start: pred_start + pred_lenth]
            Y_ = self.Y[train_start: pred_start + pred_lenth]
            print("X_ lenth: ", len(X_))
            #         zscore标准化
            X_ = stats.zscore(X_)
            #         对称正交
            X_ = self.Symmetry(X_)

            x_train = X_[train_start: train_start + train_lenth]
            y_train = Y_[train_start: train_start + train_lenth]
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=41)
            x_pred = self.X[pred_start: pred_start + pred_lenth]

            return x_train, y_train, x_test, y_test, x_pred

      def gbm_eval(self, x_train, y_train, x_test, y_test, x_pred):
            """

            :param x_train:
            :param y_train:
            :param x_test:
            :param y_test:
            :param x_pred:
            :return:
            """
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
            params = {
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': {'l1', 'l2'},  # l1和l2代表两种误差计算
                  'num_leaves': 30,
                  'max_depth': 5,
                  'learning_rate': 0.05,
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.8,
                  'bagging_freq': 5,
                  'verbose': -1,
                  'n_estimators': 1000,
            }

            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=100,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=30,
                            )

            y_pred = gbm.predict(x_pred, num_iteration=gbm.best_iteration)

            return y_pred


      def iter(self):

            i = 0
            while 1:
                  if len(self.y_pred_full) >= len(self.X):
                        break

                  x_train, y_train, x_test, y_test, x_pred = self.init(i)

                  y_pred = self.gbm_eval(x_train, y_train, x_test, y_test, x_pred)

                  self.y_pred_full += list(y_pred)
                  print("y_pred_full lenth: ", len(self.y_pred_full))

                  i += 1

            self.X['y_pred_full'] = self.y_pred_full


      def stat(self, to_excel=0):
            """

            :return:
            """
            reg_dict = {
                  'Date': list(self.data['Date']),
                  'close': list(self.data['close']),
                  'realRes': list(self.Y),
                  'pred': list(self.X['y_pred_full'])
            }

            stat_dict = {
                  "（真实情况）敲出 - 预测准确率" : None,
                  "（真实情况）敲入 - 预测准确率" : None,
                  "（预测情况）敲出 - 预测准确率" : None,
                  "（预测情况）敲入 - 预测准确率" : None,
                  "（真实情况）敲出占比" : None,
                  "预测敲出数 / 真实敲出数" : None,
                  "预测敲入数 / 真实敲入数" : None,
            }

            df = pd.DataFrame(reg_dict)[self.train_len + 252:]
            # df.to_excel('temp.xlsx', index=False)

            realRes = list(df['realRes'])
            pred = list(df['pred'])
            ths = 0.7
            KickOUT_total = 0
            KickIN_total = 0
            KickOUT_win = 0
            KickIN_win = 0
            KickOUT_loss = 0
            KickIN_loss = 0

            for i in range(len(realRes)):
                  if realRes[i] == 1.0:
                        KickOUT_total += 1
                        if pred[i] >= ths:
                              KickOUT_win += 1
                        else:
                              KickOUT_loss += 1
                  if realRes[i] == 0.5:
                        KickOUT_total += 1
                        if pred[i] >= ths:
                              KickOUT_win += 1
                        else:
                              KickOUT_loss += 1
                  elif realRes[i] == 0.0:
                        KickIN_total += 1
                        if pred[i] < ths:
                              KickIN_win += 1
                        else:
                              KickIN_loss += 1

            stat_dict["（真实情况）敲出 - 预测准确率"] = str(KickOUT_win / KickOUT_total)
            stat_dict["（真实情况）敲入 - 预测准确率"] = str(KickIN_win / KickIN_total)
            stat_dict["（真实情况）敲出占比"] = str(KickOUT_total / len(realRes))

            # 先知道模拟结果再求胜率
            realRes = list(df['realRes'])
            pred = list(df['pred'])
            ths = 0.7
            KickOUT_total = 0
            KickIN_total = 0
            KickOUT_win = 0
            KickIN_win = 0
            KickOUT_loss = 0
            KickIN_loss = 0

            for i in range(len(realRes)):
                  if pred[i] >= ths:
                        KickOUT_total += 1
                        if realRes[i] == 1:
                              KickOUT_win += 1
                        elif realRes[i] == 0.5:
                              KickOUT_win += 1
                        else:
                              KickOUT_loss += 1
                  elif pred[i] < ths:
                        KickIN_total += 1
                        if realRes[i] == 0:
                              KickIN_win += 1
                        else:
                              KickIN_loss += 1

            stat_dict["（预测情况）敲出 - 预测准确率"] = str(KickOUT_win / KickOUT_total)
            stat_dict["（预测情况）敲入 - 预测准确率"] = str(KickIN_win / KickIN_total)

            real_KO = [item for item in realRes if item >= 0.5]
            real_KI = [item for item in realRes if item < 0.5]
            # print(len(real_KO))
            stat_dict["预测敲出数 / 真实敲出数"] = KickOUT_total / len(real_KO)
            stat_dict["预测敲入数 / 真实敲入数"] = KickIN_total / len(real_KI)
            stat_df = pd.DataFrame(stat_dict)
            pprint(stat_df)

            if to_excel:
                  stat_df.to_excel("stat.xlsx")

      def plot(self):

            plt.figure(figsize=(12, 8))

            sns.lineplot(data=self.X,
                         x=self.X.index,
                         y=self.Y)

            sns.lineplot(data=self.X,
                         x=self.X.index,
                         y=self.X['y_pred_full'])

            sns.lineplot(data=self.X,
                         x=self.X.index,
                         y=0.7,
                         )


      def run(self):

            self.loadData()

            self.dataPre(self.data)

            self.iter()

            self.plot()

            self.stat()
