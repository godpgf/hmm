from pyhmm import *
from pyalphatree import *
from stdb import *
from matplotlib import pyplot as plt

colors = [(0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0),
          (1.0, 0.0, 0.0)]

TEST_DAYS = 750

if __name__ == '__main__':
    #下载股票数据
    download_industry([],'399005','data')
    print("finish download")
    #将hmm隐藏状态写入csv
    # paint_bayeshmm_hidestate("data/1399005.csv", delta_days=10)
    paint_binomialhmm_hidestate("data/399005.csv", delta_days=10)
    print("finish paint")
    #将csv变成二进制
    cache_base('data', titles=["date", "open", "high", "low", "close", "volume", "returns", "state"])
    #读取日期数据
    dataProxy = LocalDataProxy('data', True)
    data = dataProxy.get_all_data('399005')
    date = data['date']
    date = [("%d"%int(d/1000000)) for d in date]
    date = ["%s-%s-%s"%(d[:4],d[4:6],d[6:])for d in date]

    with AlphaForest() as af:
        af.load_db('data')
        label = AlphaArray('399005',["v = state"],'v',0,TEST_DAYS)[:]
        close = AlphaArray('399005',["v = close"], "v", 0, TEST_DAYS)[:]

        plt.figure(figsize=(24, 6))
        for i in range(TEST_DAYS):
            price = close[i]
            cur_date = date[i - TEST_DAYS]
            cur_hide_state = int(label[i])
            plt.plot_date([cur_date], [price], '.', color=colors[cur_hide_state])
        plt.show()
        print("finish")