from __future__ import division
import statsmodels.tsa.stattools as stattools
# acf = stattools.acf(ts,adjusted=True)
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from bayesian_changepoint_detection.generate_data import generate_normal_time_series

import numpy as np
import scipy
import seaborn
from matplotlib import pyplot as plt
from bayesian_changepoint_detection.priors import const_prior
from functools import partial
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll


def percentile_of_length(length_of_segment_c,is_coarse_grained=False):
    if not is_coarse_grained:
        print('nums of segment:' + str(len(length_of_segment_c)))
        print('minimal length of segment:' + str(np.min(length_of_segment_c)))
        print('5th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 5)))
        print('15th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 15)))
        print('25th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 25)))
        print('50th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 50)))
        print('75th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 75)))
        print('95th percentile of length of segment:' + str(np.percentile(length_of_segment_c, 95)))
        print('mean length of segment:' + str(np.mean(length_of_segment_c)))
    else:
        print('nums of coarse_grained_sequence:' + str(len(length_of_segment_c)))
        print('minimal length of coarse_grained_sequence:' + str(np.min(length_of_segment_c)))
        print('5th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 5)))
        print('15th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 15)))
        print('25th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 25)))
        print('50th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 50)))
        print('75th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 75)))
        print('95th percentile of length of coarse_grained_sequence:' + str(np.percentile(length_of_segment_c, 95)))
        print('mean length of coarse_grained_sequence:' + str(np.mean(length_of_segment_c)))


def cdf(queue, bins, title, ylable, xlable,acf=False):
    res_freq = scipy.stats.relfreq(queue, numbins=bins)
    cdf = np.cumsum(res_freq.frequency)
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    if acf:
        plt.xlim((-1,1))
    plt.grid()
    plt.plot(x, cdf)
    plt.title(title)
    plt.ylabel(ylable)
    plt.xlabel(xlable)
    plt.show()


def load_np():
    temp_file = []
    for file in os.listdir('segment_from_the_same_queue'):
        temp = []
        temp.append(int(file.replace('.npy', '').split('+')[0]))
        temp.append(int(file.replace('.npy', '').split('+')[1]))
        temp_file.append(temp)
    temp_file.sort()
    t = [temp[0] for temp in temp_file]
    l = max(t) + 1
    for i in range(l):
        segment_from_the_same_queue.append([])
    for temp in temp_file:
        segment_from_the_same_queue[temp[0]] = []
    for temp in temp_file:
        temp_seg = np.load('segment_from_the_same_queue/' + str(temp[0]) + '+' + str(temp[1]) + '.npy')
        segment_from_the_same_queue[temp[0]].append(temp_seg)

    for file in os.listdir('segment_period'):
        segment_period.append(np.load('segment_period/' + file))
    for file in os.listdir('no_segment_period'):
        no_segment_period.append(np.load('no_segment_period/' + file))


def save_np():
    for i in range(len(segment_from_the_same_queue)):
        for j in range(len(segment_from_the_same_queue[i])):
            np.save('segment_from_the_same_queue/' + str(i) + '+' + str(j) + '.npy', segment_from_the_same_queue[i][j])
    for i in range(len(segment_period)):
        np.save('segment_period/' + str(i) + '.npy', segment_period[i])
    for i in range(len(no_segment_period)):
        np.save('no_segment_period/' + str(i) + '.npy', no_segment_period[i])


files = os.listdir("lab")
# 统计得到的分段
segment_period = []
# 不能分段的部分
no_segment_period = []
num_of_trace_cannot_segment = 0
# 记录同一序列中的segment
segment_from_the_same_queue = []
j = 0
'''
for file in files:
    data = np.loadtxt("lab/"+file,usecols=[1])
    prior_function = partial(const_prior, p=1/(len(data) + 1))
    Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(),truncate=-40)
    #fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
    #ax[0].plot(data[:])
    split = np.exp(Pcp).sum(0)
    #ax[1].plot(np.exp(Pcp).sum(0))
    #plt.savefig("pics/"+file+".svg")
    #acf = stattools.acf(data,nlags=data.size)
    #ax[2].plot(acf)
    #plt.show()

    #查看突变点所在的位置。
    #print(np.where(split>0.7))
    start = 0
    end = data.size
    location_of_split=list(np.where(split>0.7)[0])
    #无法分段
    if not location_of_split:
        num_of_trace_cannot_segment+=1
        no_segment_period.append(data)
    else:
        segment_from_the_same_queue.append([])
        segment_from_the_same_queue[j]=[]
        for i in range(len(location_of_split)):
            segment_period.append(data[start:location_of_split[i]+1])
            segment_from_the_same_queue[j].append(data[start:location_of_split[i]+1])
            start=location_of_split[i]+1
        segment_period.append(data[start:end])
        segment_from_the_same_queue[j].append(data[start:end])
        j+=1

    print("循环中")


save_np()
'''
load_np()

length_of_segment = []
for i in segment_period:
    length_of_segment.append(len(i))
percentile_of_length(length_of_segment)
del length_of_segment

# 剔除分段长度小于10的段
i = 0
while i < len(segment_period):
    if segment_period[i].size < 27:
        del segment_period[i]
    else:
        i += 1

length_of_segment = []
acf_of_segment = []
for i in segment_period:
    length_of_segment.append(len(i))
    acf_of_segment.append(stattools.acf(i, nlags=26)[1:])

for i in no_segment_period:
    acf_of_segment.append(stattools.acf(i, nlags=26)[1:])

# 画CDF
cdf(length_of_segment, np.max(length_of_segment), 'CDF of length_of_segment', 'CDF', 'length(second)')
# 画直方图
fig = plt.figure(figsize=(25, 12))
ax = fig.add_subplot(111)
nums, bins, patches = plt.hist(length_of_segment, np.max(length_of_segment) // 10)
int_bin = [int(it) for it in bins]
plt.xticks(bins, int_bin)
for num, bin in zip(nums, bins):
    plt.annotate(num, xy=(bin, num), xytext=(bin + 1.5, num + 0.5))
# plt.title('hist of length_of_segment')
plt.ylabel('times of showing up')
plt.xlabel('length(second)')
plt.show()

print('after delete')
percentile_of_length(length_of_segment)
# 画分布图
sorted_data = np.sort(np.array(length_of_segment))
plt.scatter(np.arange(0, sorted_data.size, 1), sorted_data)
plt.title("distribution of sorted length_of_segment")
plt.ylabel('length(seconds)')
plt.show()
# length_cdf=scipy.stats.norm.cdf(length_of_segment)
# #print(length_of_segment)
# seaborn.lineplot(x=np.array(length_of_segment),y=np.array(length_cdf))
acf = np.array(acf_of_segment).T
legend=[]
percent = [[], [], [], [], []]
percent_nums = [5, 25, 50, 75, 95]
for i in range(acf.shape[0]):
    for j in range(len(percent)):
        percent[j].append(np.percentile(acf[i], percent_nums[j]))
for j in percent:
    line, = plt.plot(j)
    legend.append(line)
plt.xlabel('delays')
plt.ylabel('ACF')
plt.title('ACF of All')
plt.legend(handles=legend,labels=[str(i)+'th percentile' for i in percent_nums],loc='best')
plt.show()

# 画前5拍的cdf
for i in range(5):
    cdf(acf[i], 19, 'cdf of ' + str(i + 1) + ' delay', 'cdf', 'acf',True)

# 计算segment均值，得到粗粒度的序列
coarse_grained_sequence=[]
for i in range(len(segment_from_the_same_queue)):
    coarse_grained_sequence.append([])
    for j in range(len(segment_from_the_same_queue[i])):
        coarse_grained_sequence[i].append(np.mean(segment_from_the_same_queue[i][j]))
# 分析新的到的序列的acf
percentile_of_length([len(i) for i in coarse_grained_sequence],True)
# 计算ACF
acf_of_coarse_grained_sequence = []
for i in range(len(coarse_grained_sequence)):
    acf_of_coarse_grained_sequence.append(stattools.acf(coarse_grained_sequence[i])[1:])
    print('acf of '+str(i)+'th coarse grained sequence: '+str(acf_of_coarse_grained_sequence[i]))
print()