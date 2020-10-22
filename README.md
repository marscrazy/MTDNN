# MTDNN
This is implementation of Multi-scale Two-way Deep Neural Network [paper](https://www.ijcai.org/Proceedings/2020/0628.pdf)

# Benchmark dataset CSI-2016
> CSI-2016 is our collected dataset from three one-minute
stock index data, including the Shanghai Stock Exchange
(SSE) Composite Index SH000001, Shenzhen Stock Exchange Small & Medium Enterprises (SME Boards) Price
Index SZ399005 and ChiNext Price Index SZ399006. It
has over 170, 000 samples spanning a year from January
1st, 2016, to December 30th, 2016. Each sample is a one minute data of
6 dimensions which are high, low, open, close, volume and
amount, respectively.R

# TODO
- [] code for xgboost
- [] code for CNN-based model
- [] code for MTDNN

# esults on CSI-2016

![Image text](https://github.com/marscrazy/MTDNN/blob/master/pics/csi2016.png)

![Image text](https://github.com/marscrazy/MTDNN/blob/master/pics/csi2016-2.png)

# Citation
Please cite this paper if you use CSI-2016.
```
@inproceedings{DBLP:conf/ijcai/LiuMSHGLSLW20,
  author    = {Guang Liu an的Yuzhao Mao and Qi Sun and Hailong Huang and Weiguo Gao and Xuan Li and
    Jianping Shen and Ruifan Li and Xiaojie Wang},
  editor    = {Christian Bessiere},
  title     = {Multi-scale Two-way Deep Neural Network for Stock Trend Prediction},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2020},
  pages     = {4555--4561},
  publisher = {ijcai.org},
  year      = {2020},
  url       = {https://doi.org/10.24963/ijcai.2020/628},
  doi       = {10.24963/ijcai.2020/628},
  timestamp = {Mon, 20 Jul 2020 12:38:52 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/LiuMSHGLSLW20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```