from ctypes import *
from pyhmm.libhmm import hmm
import numpy as np

class BinomialHMM(object):

    def __init__(self, hide_state_cnt, be_cnt):
        self.cxx_p = hmm.createBinomialHMM(c_int32(hide_state_cnt), c_int32(be_cnt))
        self.cache_size = 0
        self.hide_state_cnt = hide_state_cnt

    def __del__(self):
        hmm.cleanHMM(c_void_p(self.cxx_p))

    def train(self, x, epoch_num = 8):
        self._init_cache(x)
        hmm.trainHMM(c_void_p(self.cxx_p), self.cache, c_int32(1), c_int32(len(x)), c_int32(epoch_num))

    def _init_cache(self, x):
        cache_size = len(x)
        if self.cache_size < cache_size:
            self.cache_size = cache_size
            self.cache = (c_float * cache_size)()
        for id, cur_x in enumerate(x):
            self.cache[id] = float(cur_x)

    def get_hide_state(self):
        return hmm.getCurrentHideState(c_void_p(self.cxx_p))

    def get_hide_state_coff(self):
        if self.cache_size < self.hide_state_cnt:
            self.cache_size = self.hide_state_cnt
            self.cache = (c_float * self.hide_state_cnt)()
        hmm.getHideStateCoff(c_void_p(self.cxx_p), self.cache)
        return np.array([self.cache[i] for i in range(self.hide_state_cnt)])