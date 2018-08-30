//
// Created by 严宇 on 2018/8/31.
//

#ifndef HMM_BAYESHMM_H
#define HMM_BAYESHMM_H

#include "basehmm.h"

class BayesHMM : public BayesHMM{
public:
    BayesHMM(unsigned n, unsigned m):BayesHMM(n, m){}

protected:
    virtual float getOutProbability(unsigned n, unsigned t, DMatrix<float>* o){
        return (*model_.getB())(n, (unsigned)o->get(t,0));
    }

    virtual void updateB(DMatrix<float>* gamma, DMatrix<float>* o, DMatrix<float>* b){
        for(unsigned i = 0; i < b->getDim1(); ++i){
            float sumGamma = 0;

            for(unsigned t = 0; t < o->getDim1(); ++t){
                sumGamma += (*gamma)(t, i);
            }
            for(unsigned k = 0; k < b->getDim2(); ++k){
                float sumMaskGamma = 0;
                for(unsigned t = 0; t < o->getDim1(); ++t){
                    if((unsigned)(*o)(t,0) == k){
                        sumMaskGamma += (*gamma)(t, i);
                    }
                }
                (*b)(i, k) = sumMaskGamma / sumGamma;
            }
        }
    }
};

#endif //HMM_BAYESHMM_H
