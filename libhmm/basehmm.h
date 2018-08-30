//
// Created by godpgf on 18-8-30.
//

#ifndef HMM_BASEHMM_H
#define HMM_BASEHMM_H

#include "base/matrix.h"

class HMMModel {
public:
    HMMModel(unsigned n, unsigned m) : n_(n), m_(m) {
        a_ = new DMatrix<float>(n, n);
        b_ = new DMatrix<float>(n, m);
        pi_ = new DMatrix<float>(n, 1);
    }


    virtual ~HMMModel() {
        delete a_;
        delete b_;
        delete pi_;
    }

    DMatrix<float> *getA() { return a_; }

    DMatrix<float> *getB() { return b_; }

    DMatrix<float> *getPi() { return pi_; }


protected:
    //隐藏状态的数量
    unsigned n_;
    //观测到的行为数量 或者 观测到的概率分布的参数数量（正态分布是2）
    unsigned m_;
    //状态之间的转移概率
    DMatrix<float> *a_;
    //状态到可观测行为的转移概率 或可观测行为的概率分布参数
    DMatrix<float> *b_;
    //初始状态概率
    DMatrix<float> *pi_;
};

class BaseHMM{
public:
    BaseHMM(unsigned n, unsigned m):model_(n,m){

    }

protected:
    //模型参数
    HMMModel model_;
    //前向概率
    DMatrix<float> alpha_;
    //后向概率
    DMatrix<float> beta_;
    //t时刻处于某个状态的概率
    DMatrix<float> gamma_;
    //t时刻处于状态i，t+1时刻处于状态j的概率
    DMatrix<float> xi_;
};

#endif //HMM_BASEHMM_H
