//
// Created by godpgf on 18-8-30.
//

#ifndef HMM_BASEHMM_H
#define HMM_BASEHMM_H

#include "base/matrix.h"
#include <math.h>
using namespace std;

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

    void train(DMatrix<float> *o, int epochNum) {
        init(o);
        do {
            doForwardPass(o);
            doBackwardPass(o);
            computePosteriors();
            calXi(o);
            updatePi(&gamma_, model_.getPi());
            updateA(&xi_, &gamma_, model_.getA());
            updateB(&gamma_, o, model_.getB());
        } while (--epochNum);
    }

    //得到第n步隐藏状态是n时观测输出值o[t]的发生概率
    virtual float getOutProbability(unsigned n, unsigned t, DMatrix<float> *o) {
        return 0.0f;
    }

    //初始化
    virtual void init(DMatrix<float> *o) {
        //初始化中间结果需要的内存
        alpha_.setSize(o->getDim1(), model_.getA()->getDim1());
        beta_.setSize(o->getDim1(), model_.getA()->getDim1());
        scale_.setSize(o->getDim1(), 1);

        //初始化隐藏状态间的转移概率
        for (unsigned i = 0; i < model_.getA()->getDim1(); ++i) {
            float sum = 0;
            for (unsigned j = 0; j < model_.getA()->getDim2(); ++j) {
                (*model_.getA())(i, j) = rand() / float(RAND_MAX);
                sum += (*model_.getA())(i, j);
            }
            for (unsigned j = 0; j < model_.getA()->getDim2(); ++j){
                (*model_.getA())(i, j) /= sum;
            }

        }

        //初始化最开始隐藏状态发生的概率
        float sum = 0;
        for (unsigned i = 0; i < model_.getPi()->getDim1(); ++i) {
            (*model_.getPi())(i, 0) = rand() / float(RAND_MAX);
            for (unsigned j = 0; j < model_.getB()->getDim2(); ++j)
                sum += (*model_.getPi())(i, j);
        }
        for (unsigned i = 0; i < model_.getPi()->getDim1(); ++i) {
            (*model_.getPi())(i, 0) /= sum;
        }
    }

    virtual void doForwardPass(DMatrix<float> *o) {
        scale_.init(0);
        for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
            alpha_(0, i) = (*model_.getPi())(i, 0) * getOutProbability(i, 0, o);
            scale_(0, 0) += alpha_(0, i);
        }
        scale_(0,0) = 1.0f / scale_(0,0);
        for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
            alpha_(0, i) *= scale(0,0);
        }
        for (unsigned t = 1; t < o->getDim1(); ++t) {
            for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
                float pi = 0;
                for (unsigned j = 0; j < alpha_.getDim2(); ++j)
                    pi += alpha_(t - 1, j) * (*model_.getA())(j, i);
                alpha_(t, i) = pi * getOutProbability(i, t, o);
                scale_(t, 0) += alpha_(t, i);
            }
            scale_(t, 0) = 1.0f / scale_(t, 0);
            for (unsigned i = 0; i < alpha_.getDim2(); ++i)
                alpha_(t, i) *= scale_(t, 0);
        }
    }

    virtual void doBackwardPass(DMatrix<float> *o) {
        for (unsigned i = 0; i < beta_.getDim2(); ++i) {
            beta_(-1, i) = scale_(-1, 0);
        }
        for (int t = beta_.getDim1() - 2; t >= 0; --t) {
            for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
                float pi = 0;
                for (unsigned j = 0; j < alpha_.getDim2(); ++j)
                    pi += beta_(t + 1, j) * (*model_.getA())(i, j) * getOutProbability(j, t + 1, o);
                beta_(t, i) = pi * scale_(t, 0);
                //cout << "beta[" << t << "][" << i << "]=" << beta_(t, i) << endl;
            }
        }
    }

    virtual void computePosteriors(){
        for (unsigned t = 0; t < gamma_.getDim1(); ++t) {
            float pi = 0;
            for (unsigned j = 0; j < gamma_.getDim2(); ++j)
                pi += alpha_(t, j) * beta_(t, j);
            for (unsigned i = 0; i < gamma_.getDim2(); ++i) {
                gamma_(t, i) = alpha_(t, i) * beta_(t, i) / pi;
                //cout << "gamma[" << t << "][" << i << "]=" << gamma_(t, i) << endl;
            }

        }
    }

    virtual void calXi(DMatrix<float> *o) {
        for (unsigned t = 0; t < xi_.getDim1(); ++t) {
            float pi = 0;
            for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
                for (unsigned j = 0; j < beta_.getDim2(); ++j) {
                    pi += alpha_(t, i) * (*model_.getA())(i, j) * getOutProbability(j, t + 1, o) * beta_(t + 1, j);
                }
            }

            for (unsigned i = 0; i < alpha_.getDim2(); ++i) {
                for (unsigned j = 0; j < beta_.getDim2(); ++j) {
                    xi_(t, i * alpha_.getDim2() + j) =
                            alpha_(t, i) * (*model_.getA())(i, j) * getOutProbability(j, t + 1, o) * beta_(t + 1, j) /
                            pi;
                }
            }
        }
    }

    void updatePi(DMatrix<float> *gamma, DMatrix<float> *pi) {
        for (unsigned i = 0; i < pi->getDim1(); ++i) {
            (*pi)(i, 0) = (*gamma)(0, i);
        }
    }

    void updateA(DMatrix<float> *xi, DMatrix<float> *gamma, DMatrix<float> *a) {

        for (unsigned i = 0; i < a->getDim1(); ++i) {
            float sumGamma = 0;
            for (unsigned t = 0; t < xi->getDim1(); ++t) {
                sumGamma += (*gamma)(t, i);
            }

            for (unsigned j = 0; j < a->getDim2(); ++j) {
                float sumXi = 0;
                for (unsigned t = 0; t < xi->getDim1(); ++t) {
                    sumXi += (*xi)(t, i * a->getDim1() + j);
                }
                (*a)(i, j) = sumXi / sumGamma;
            }
        }

    }

    virtual void updateB(DMatrix<float> *gamma, DMatrix<float> *o, DMatrix<float> *b) {}

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
    //防止浮点数下溢
    DMatrix<float> scale_;
};

#endif //HMM_BASEHMM_H
