//
// Created by godpgf on 18-8-31.
//

#include "binomialhmm.h"
#include "bayeshmm.h"
#include "gaussianhmm.h"

#ifndef WIN32 // or something like that...
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif


extern "C"
{

/*
 * 二项分布的隐马尔科夫
 * 隐藏状态是n，beCnt次伯努利实验
 * */
void* DLLEXPORT createBinomialHMM(int n, int beCnt){
    auto p = (void*)(new BinomialHMM(n, beCnt));
    return p;
}

/*
 * 离散贝叶斯因马尔科夫（最简单的）
 * n是隐藏状态，m是事件数量
 * */
void* DLLEXPORT createBayesHMM(int n, int m){
    auto p = (void*)(new BayesHMM(n, m));
    return p;
}

/*
 * 高斯分布因马尔科夫（最难，以后实现，占时不需要用）
 * n是隐藏状态，m是数据维度
 * */
void* DLLEXPORT createGaussianHMM(int n, int m){
    auto p = (void*)(new GaussianHMM(n, m));
    return p;
}

void DLLEXPORT trainHMM(void* p, float* x, int dim, int len, int epochNum){
    DMatrix<float> o(len, dim, x);
    BaseHMM* hmm = (BaseHMM*)p;
    hmm->train(&o, epochNum);
}


int DLLEXPORT getCurrentHideState(void* p){
    return ((BaseHMM*)p)->getCurrentHideState();
}

int DLLEXPORT getHideStateCoff(void* p, float* outCoff){
    return ((BaseHMM*)p)->getHideStateCoff(outCoff);
}

void DLLEXPORT cleanHMM(void* p){
    delete ((BaseHMM*)p);
}

}
