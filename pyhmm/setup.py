#coding=utf-8
#author=godpgf
import platform
from setuptools import setup, find_packages

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

sysstr = platform.system()


setup(
    name='pyhmm',
    version='0.0.1',
    description='libHMM for python',
    packages=find_packages(exclude=[]),
    author='godpgf',
    author_email='godpgf@qq.com',
    package_data={'': ['*.*']},
    data_files = [('lib',[('../lib/libhmm_api.dll' if sysstr =="Windows" else '../lib/libhmm_api.so')])],
    url='https://github.com/godpgf',
    install_requires=[str(ir.req) for ir in parse_requirements("requirements.txt", session=False)],
    zip_safe=False,
    #entry_points={
    #    "console_scripts": [
    #        "rqalpha = rqalpha.__main__:entry_point",
    #    ]
    #},
)