//
// Created by godpgf on 18-8-31.
//

#ifndef HMM_GAUSSIANHMM_H
#define HMM_GAUSSIANHMM_H

#include "basehmm.h"

class GaussianHMM : public BaseHMM{
public:
    GaussianHMM(unsigned n, unsigned m):BaseHMM(n, m){

    }

protected:
    virtual float getOutProbability(unsigned n, unsigned t, DMatrix<float>* o){
        //todo
        return 0;
    }
    virtual void init(DMatrix<float>* o){
        //todo
    }
    virtual void updateB(DMatrix<float>* gamma, DMatrix<float>* o, DMatrix<float>* b){
        //todo
    }
    static const float _pi;
};

const float GaussianHMM::_pi = 1.f / sqrtf(M_PI * 2);

#endif //HMM_GAUSSIANHMM_H
