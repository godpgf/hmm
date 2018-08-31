//
// Created by godpgf on 18-8-31.
//

#ifndef HMM_BINOMIALHMM_H
#define HMM_BINOMIALHMM_H

#include "basehmm.h"

class BinomialHMM : public BaseHMM{
public:
    //beCnt:伯努利实现次数
    BinomialHMM(unsigned n, unsigned beCnt):BaseHMM(n,1),beCnt_(beCnt), cnk_(beCnt + 1, beCnt + 1){
        cnk_.init(-1.0);
        for(int i = 0; i <= beCnt; ++i){
            for(int j = 0; j <= i; ++j){
                if(i == j || j == 0){
                    cnk_(i, j) = 1;
                }
            }
        }
        for(int k = 0; k <= beCnt; ++k){
            calCnk(beCnt, k);
            //cout<<"c"<<beCnt<<" "<<k<<" "<<calCnk(beCnt, k)<<endl;
        }
    }

protected:

    //初始化
    virtual void init(DMatrix<float> *o) {
        BaseHMM::init(o);

        for (unsigned i = 0; i < model_.getB()->getDim1(); ++i) {
            for (unsigned j = 0; j < model_.getB()->getDim2(); ++j) {
                (*model_.getB())(i, j) = rand() / float(RAND_MAX);
            }
        }
    }
    virtual float getOutProbability(unsigned n, unsigned t, DMatrix<float>* o){
        int k = (int)o->get(t, 0);
        float p = (*model_.getB())(n, 0);
//        cout<<beCnt_<<" "<<k<<" "<<p<<" "<<k<<endl;
//        cout<<"out p:"<< cnk_(beCnt_, k)<<" "<<powf(p, k)<<" "<<powf(1-p,n-k)<<endl;
        return cnk_(beCnt_, k) * powf(p, k) * powf(1-p,beCnt_-k);
    }

    int calCnk(unsigned n, unsigned k){
        if(cnk_(n, k) > 0)
            return cnk_(n, k);
        cnk_(n, k) = calCnk(n-1, k-1) + calCnk(n-1, k);
        return cnk_(n, k);
    }

    //参考文献《解密复兴科技：基于隐蔽马尔科夫模型的时许分析方法》p81，原文没有二项分布，需要需要自己推导
    virtual void updateB(DMatrix<float>* gamma, DMatrix<float>* o, DMatrix<float>* b){
        for(unsigned i = 0; i < b->getDim1(); ++i){
            float sumGammaUp = 0, sumGammaDown = 0;

            for(unsigned t = 0; t < o->getDim1(); ++t){
                sumGammaUp += (*gamma)(t, i) * (*o)(t,0);
                sumGammaDown += (*gamma)(t, i) * (beCnt_ - (*o)(t,0));
            }
//            cout<<sumGammaUp<<"/"<<sumGammaDown<<endl;
            float g = sumGammaUp / sumGammaDown;
            (*b)(i, 0) = g / (1 + g);
        }
    }

    unsigned beCnt_;
    DMatrix<int> cnk_;
};

#endif //HMM_BINOMIALHMM_H
