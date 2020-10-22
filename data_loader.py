from sklearn.preprocessing import StandardScaler
from datetime import datetime
from path import Path
import numpy as np
import pywt
import math

DATA_DIR = "data"
def transform(x):
    db1 = pywt.Wavelet('haar')
    rs = pywt.wavedec(x,db1)
    rs = np.concatenate(rs)
    return rs

def trans4d(x):
    x = x.reshape(100,-1)
    new_x = [transform(x[:,i]) for i in range(x.shape[1])]
    return np.concatenate(new_x)

def str2time(s):
    return datetime.strptime(s, '%Y/%m/%d %H:%M')

def read_data(name='SH1A0001'):
    results = []
    time_set = set()
    for filename in Path(DATA_DIR).files():
        # print(filename)
        if name in filename.name:
            count = 0
            with open(filename, 'r', encoding='utf8') as infile:
                for line in infile:
                    line = line.strip().split(',')
                    d, t, o, h, l, c, v, m = line
                    o, h, l, c, v, m = list(map(float, [o, h, l, c, v, m]))
                    dt = str2time(d + ' ' + t)
                    if dt not in time_set:
                        results.append([dt, [o, h, l, c, v, m]])
                        time_set.add(dt)
                    if m == v == 0:
                        count += 1
            #print(filename, ":", count)
    results.sort(key=lambda x: x[0])
    return results

def dataset_gen(name, win=10, lag=1):
    print("reading dataset {}-{}-{}".format(name, win, lag))
    dataset = read_data(name)
    raw_data = [x[1] for x in dataset if x[1][3]!=0]
    sc = StandardScaler()
    raw_data = sc.fit_transform(raw_data)
    new_dataset = []
    new_labels = []
    for i in range(win - 1, len(raw_data) - lag):
        tmp = raw_data[i - win + 1:i + 1]
        new_dataset.append(tmp.reshape(-1))
        new_labels.append( (np.mean(raw_data[i+1:i+lag+1,3]) - raw_data[i,3])/raw_data[i,3])
    return np.array(new_dataset), np.array(new_labels)

def label_th(x, th):
    if x < -th:
        return 0
    elif x > th:
        return 2
    else:
        return 1

def get_dataset(name="0001", win=100, lag=1, th=0.01):
    X, Y = dataset_gen(name, win, lag)
    X_train, X_test = X[:-10000], X[-10000:]
    Y_train, Y_test = Y[:-10000], Y[-10000:]
    Y_train = [label_th(z,th) for z in Y_train]
    Y_test = [label_th(z,th) for z in Y_test]
    return np.array(X_train,dtype=np.float32),Y_train, np.array(X_test,dtype=np.float32), Y_test

def get_CSI2016():
    win = 100
    lag = 5
    th = 0.01
    X_train1, Y_train1, X_test1, Y_test1 = get_dataset('001', win=win, lag=lag,th=th)
    X_train2, Y_train2, X_test2, Y_test2 = get_dataset('005', win=win, lag=lag, th=th)
    X_train3, Y_train3, X_test3, Y_test3 = get_dataset('006', win=win, lag=lag, th=th)
    X_train = np.concatenate([X_train1, X_train2, X_train3],0)
    Y_train = np.concatenate([Y_train1, Y_train2, Y_train3], 0)
    X_test = np.concatenate([X_test1, X_test2, X_test3], 0)
    Y_test = np.concatenate([Y_test1, Y_test2, Y_test3],0)
    X_train = np.array([trans4d(x) for x in X_train])
    X_test = np.array([trans4d(x) for x in X_test])

    dsize, length = X_train.shape
    idx = list(range(dsize))
    np.random.shuffle(idx)
    X_train = X_train[idx]
    Y_train = Y_train[idx]
    return X_train, Y_train, X_test, Y_test

def states(Y):
    print("{},{},{}".format(sum([1 for x in Y if x==0]),
                            sum([1 for x in Y if x==1]),
                            sum([1 for x in Y if x==2])))
if __name__=="__main__":
    X_train,Y_train,X_test,Y_test = get_CSI2016()
    states(Y_test)
    states(Y_train)