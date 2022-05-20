from MonteCarlo import MCS
from datetime import datetime
from time import time

if __name__ == '__main__':
    print("启动模拟时间：", datetime.now())
    # 计时开始
    tic = time()
    # 初始化
    mcs = MCS(times=int(1e+4), mode='sim', type='future', year=7/12)
    # 运行蒙特卡洛模拟
    snowKick = mcs.run()
    # 计时结束
    tok = time()
    snowKick.to_excel('snowKick.xlsx')
    # 显示结果
    print(snowKick)
    print("运行时间：", round(tok - tic, 2), 's')