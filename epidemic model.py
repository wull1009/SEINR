# 1. SEIR 模型，常微分方程组
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def dySEINR(y, t, lamda1, lamda2, sigma, gama, delta, mu, yita):  # SEINR 模型，导数函数
    s, e, i, n = y  # youcans
    ds_dt = sigma * n + (1 - gama) * mu * i - lamda1 * s * e - lamda2 * s * n
    de_dt = delta * (lamda1 * s * e + lamda2 * s * n) - yita * e
    di_dt = yita * e - mu * i
    # dr_dt = gama * mu * i
    dn_dt = (1 - delta) * (lamda1 * s * e + lamda2 * s * n) - sigma * n
    return np.array([ds_dt, de_dt, di_dt, dn_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda1 = 0.4  # 日接触率, 每个患病者每天有效接触的易感者的平均人数
lamda2 = 0.2  # 日检出率，每个无症状感染者每天有效接触的易感者的平均人数。
yita = 0.07  # 日发病率，每天发病成为患病者的潜伏者占潜伏者总数的比例
delta = 0.4  # 接触后的发病率，即易感者接触欲患病和无症状者以后成为欲患病者的比例
mu = 0.05  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
sigma = 0.1  # 无症状转阴率，即每天无症状感染者变回易感者且丧失传染性的比率
gama = 0.8  # 治愈患者中变成终身免疫者的比例
sig1 = lamda1 / yita
sig2 = lamda2 / sigma
fsig1 = 1 - 1 / sig1
fsig2 = 1 - 1 / sig2
tEnd = 500  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
i0 = 0  # 患病者比例的初值
e0 = 4e-3  # 欲患病者比例的初值
n0 = 6e-3  # 无症状感染者比例的初值
s0 = 1 - i0 - n0 - e0  # 易感者比例的初值

Y0 = (s0, e0, i0, n0)  # 微分方程组的初值

# odeint 数值解，求解微分方程初值问题
ySEINR = odeint(dySEINR, Y0, t, args=(lamda1, lamda2, sigma, gama, delta, mu, yita))  # SEINR 模型
print(ySEINR[:, 0])
# 输出绘图
print("lamda1={}\tlamda2={}\tsigma={}\tgama={}\tdelta={}\tmu={}\tyita={}\t(1-1/sig1)={}\t(1-1/sig2)={}".format(lamda1, lamda2, sigma, gama, delta, mu, yita, fsig1, fsig2))
plt.title("Comparison among SI, SIS, SIR and SEIR models")
plt.xlabel('t-youcans')
plt.axis([0, tEnd, -0.1, 1.1])
# plt.plot(t, 1-ySIR[:,0]-ySIR[:,1], 'cornflowerblue', label='r(t)-SIR')
plt.plot(t, ySEINR[:, 0], '--', color='darkviolet', label='s(t)-SEINR')
plt.plot(t, ySEINR[:, 1], '-.', color='orchid', label='e(t)-SEINR')
plt.plot(t, ySEINR[:, 2], '-', color='m', label='i(t)-SEINR')
plt.plot(t, ySEINR[:, 3], '-', color='purple', label='n(t)-SEINR')
plt.plot(t, 1 - ySEINR[:, 0] - ySEINR[:, 1] - ySEINR[:, 2] - ySEINR[:, 3], ':', color='palevioletred', label='r(t)-SIENR')
plt.legend(loc='right')  # youcans
# plt.show()
plt.savefig('./传染病.png', dpi=500, bbox_inches='tight')
