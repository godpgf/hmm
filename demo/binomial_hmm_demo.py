from pyhmm import *
from pyalphatree import *
from stdb import *
from matplotlib import pyplot as plt

TEST_DAYS = 1500
HMM_TRAIN_DAYS = 300
BE_CNT = 10

colors = [(0.0, 1.0, 0.0),
          (0.0, 0.0, 1.0),
          (1.0, 0.0, 0.0)]

if __name__ == '__main__':
    #下载股票数据
    download_industry(['1399005'],'1399005','data')
    #将csv变成二进制
    cache_base('data', titles=["date", "open", "high", "low", "close", "volume", "returns"])
    #读取日期数据
    dataProxy = LocalDataProxy('data', True)
    data = dataProxy.get_all_data('1399005')
    date = data['date']
    date = [("%d"%int(d/1000000)) for d in date]
    date = ["%s-%s-%s"%(d[:4],d[4:6],d[6:])for d in date]

    with AlphaForest() as af:
        af.load_db('data')
        label = AlphaArray('1399005',["v = sum((returns > 0.004), %d)"%(BE_CNT-1)],'v',0,TEST_DAYS)[:]
        close = AlphaArray('1399005',["v = close"], "v", 0, TEST_DAYS)[:]
        hmm_model = BinomialHMM(len(colors), BE_CNT)

        plt.figure(figsize=(24, 6))
        for i in range(HMM_TRAIN_DAYS, TEST_DAYS):
            print("%d/%d"%(i,TEST_DAYS))
            price = close[i]
            cur_date = date[i - HMM_TRAIN_DAYS]
            history_lab = label[i-HMM_TRAIN_DAYS:i]
            hmm_model.train(history_lab, epoch_num=8)
            cur_hide_state = hmm_model.get_hide_state()
            cur_hide_state_coff = hmm_model.get_hide_state_coff()
            real_hide_state = 0
            coff = cur_hide_state_coff[cur_hide_state]
            for cur_c in cur_hide_state_coff:
                if cur_c < coff:
                    real_hide_state += 1
            plt.plot_date([cur_date], [price], '.', color=colors[real_hide_state])
        plt.show()
        print("finish")

