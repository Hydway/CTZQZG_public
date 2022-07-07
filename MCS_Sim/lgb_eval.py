from lgb import lgb
from datetime import datetime
from time import time

if __name__ == '__main__':
    print("启动模拟时间：", datetime.now())
    # 计时开始
    tic = time()
    # 初始化
    lgb_model = lgb()
    # 直接运行
    lgb_model.run()
    # 计时结束
    tok = time()
    print("运行时间：", round(tok - tic, 2), 's')