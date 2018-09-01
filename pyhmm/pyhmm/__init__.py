from .bayeshmm import BayesHMM
from .binomialhmm import BinomialHMM
import numpy as np


def _read_datas(path, price_column):
    with open(path, 'r') as rf:
        titles = rf.readline()[:-1].split(',')
        price_id = None
        for id, name in enumerate(titles):
            if name == price_column:
                price_id = id
                break
        datas = []
        need_to_append = (not titles[-1] == 'state')
        if need_to_append:
            titles.append('state')
        line = rf.readline()
        while line:
            line = line[:-1].split(',')
            if need_to_append:
                line.append('0')
            datas.append(line)
            line = rf.readline()
    return titles, datas, price_id


def _write_datas(path, titles, datas):
    with open(path, 'w') as wf:
        wf.write('%s\n' % (','.join(titles)))
        for data in datas:
            wf.write('%s\n' % (','.join(data)))


def _paint_bayeshmm(datas, price_id, hide_state_num, delta_days, train_days, epoch_num):
    index = 0
    min_train_days = train_days + delta_days
    close_price = []
    while index < len(datas) and (len(close_price) < min_train_days - 1 or train_days == 0):
        close_price.append(float(datas[index][price_id]))
        index += 1

    hmm_model = BayesHMM(hide_state_num, 2)
    while index < len(datas):
        close_price.append(float(datas[index][price_id]))
        cur_close = np.array(close_price[-train_days:])
        before_close = np.array(close_price[-train_days - delta_days:-delta_days])
        label = np.where(cur_close > before_close, 1.0, 0.0)
        hmm_model.train(label, epoch_num)
        cur_hide_state = hmm_model.get_hide_state()
        cur_hide_state_coff = hmm_model.get_hide_state_coff()
        real_hide_state = 0
        coff = cur_hide_state_coff[cur_hide_state][1]
        for cur_c in cur_hide_state_coff:
            if cur_c[1] < coff:
                real_hide_state += 1
        datas[index][-1] = "%d" % real_hide_state
        index += 1
    return datas


#将计算出的隐藏状态写入股票数据的csv文件中
def paint_bayeshmm_hidestate(path, hide_state_num=3, price_column='close', delta_days=1, train_days=300, epoch_num=8):
    titles, datas, price_id = _read_datas(path, price_column)
    datas = _paint_bayeshmm(datas, price_id, hide_state_num, delta_days, train_days, epoch_num)
    _write_datas(path, titles, datas)
