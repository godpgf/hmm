import ctypes
import platform
from ctypes import *

sysstr = platform.system()

import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/../../lib/"

try:
    hmm = ctypes.windll.LoadLibrary(lib_path + 'libhmm_api.dll') if sysstr =="Windows" else ctypes.cdll.LoadLibrary(lib_path + 'libhmm_api.so')
except OSError as e:
    lib_path = curr_path + "/../../../lib/"
    hmm = ctypes.windll.LoadLibrary(
        lib_path + 'libhmm_api.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libhmm_api.so')

hmm.createBinomialHMM.restype = c_void_p
hmm.createBayesHMM.restype = c_void_p
hmm.getHideStateCoff.restype = c_int32
