import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)

x0 = np.full(m, 1.0)

input_data = np.c_[(x0, x)]
target_data = 3 * x + 8 + np.random.randn(m)

# plt.scatter(x, target_data)
# plt.show()

# 两种终止条件
# max_iter = 10000
# epsilon = 1e-5
#
# # 初始化权值
# np.random.seed(0)
# w = np.random.randn(2)
# v = np.zeros(2)  # 更新的速度参数
#
# alpha = 0.0005  # 步长
# diff = 0.
# error = np.zeros(2)
# count = 0  # 循环次数
# cost = []
# eps = 0.9  # 衰减力度,可以用来调节,该值越大那么之前的梯度对现在方向的影响也越大
# while count < max_iter:
#     count += 1
#     sum_m = np.zeros(2)
#     sum_m1 = np.zeros(2)
#
#     index = random.sample(range(m), int(np.ceil(m * 0.2))) #随机选10个数据
#     sample_data = input_data[index]
#     sample_target = target_data[index]
#
#     cost.append(np.sum((np.dot(sample_data, w)-sample_target)**2, axis=0)/(2*len(sample_data)))
#     sum_m = sample_data.T.dot(np.dot(sample_data, w)-sample_target)
#
#     # w = w - alpha * sum_m
#     #
#     v = 0.9 * v + 0.1 * sum_m     # 在这里进行速度更新
#     w = w - alpha*v                       # 使用动量来更新参数
#
#     if np.linalg.norm(w - error) < epsilon:
#         break
#     else:
#         error = w
#
# print('loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1]))
#
# plt.plot(np.arange(count), cost)
# plt.ylim((0, 10))
# plt.show()
#
#
# # plt.plot(x, input_data.dot(w))
# # plt.show()

def gradient_decent(x, y, epoches, batch_size, lr, beta=0.9, type=None):
    mm, n = x.shape
    x_ = np.c_[np.ones((mm, 1)), x]
    theta = np.zeros(n+1)
    cost = []

    if type == 'normal':
        for epoch in range(epoches):
            # np.random.shuffle(data)   #?????????????????
            baeches = [[x_[i:i+batch_size], y[i:i+batch_size]] for i in range(0, mm, batch_size)]
            for x_t, y_t in baeches:
                error = x_t.dot(theta) - y_t
                cost.append(np.sum(error ** 2, axis=0) / (2 * batch_size))
                theta -= lr * x_t.T.dot(error) / batch_size

        return theta, cost

    if type == 'momentum':

        v = np.zeros(n+1)
        for epoch in range(epoches):
            # np.random.shuffle(data)   #?????????????????
            baeches = [[x_[i:i+batch_size], y[i:i+batch_size]] for i in range(0, mm, batch_size)]
            for x_t, y_t in baeches:
                error = x_t.dot(theta) - y_t
                cost.append(np.sum(error ** 2, axis=0) / (2 * batch_size))
                grad = x_t.T.dot(error) / batch_size

                v= beta* v + (1-beta)*grad
                theta -= 10*lr * v  # multiply 10 due to (1-beta)*grad
        return theta, cost

    if type == 'adam':
        theta = np.zeros(n)
        bias =0
        m = np.zeros(n)
        v = np.zeros(n+1)
        t = 0
        b1 = 0.9
        b2 = 0.999
        epsilon = 10**(-8)

        vdw = 0
        vdb = 0
        sdw = 0
        sdb = 0

        for epoch in range(epoches):
            # np.random.shuffle(data)   #?????????????????
            baeches = [[x[i:i+batch_size], y[i:i+batch_size]] for i in range(0, mm, batch_size)]
            for x_t, y_t in baeches:
                error = x_t.dot(theta) + bias - y_t
                cost.append(np.sum(error ** 2, axis=0) / (2 * batch_size))
                dw = x_t.T.dot(error) / batch_size
                db= error.sum()/batch_size

                vdw = b1*vdw+(1-b1)*dw
                vdb = b1*vdb+(1-b1)*db

                sdw = b2 * sdw + (1 - b2) * (dw ** 2)
                sdb = b2 * sdb + (1 - b2) * (db ** 2)

                t += 1

                vdwc = vdw / (1 - b1 ** t)
                vdbc = vdb / (1 - b1 ** t)

                sdwc = sdw / (1 - b2 ** t)
                sdbc = sdb / (1 - b2 ** t)

                theta -= lr * vdwc / (np.sqrt(sdwc)+epsilon)
                bias -= lr * vdbc / (np.sqrt(sdbc)+epsilon)
        return theta,bias, cost

def data_plot(x, y, theta):
    x_new = np.linspace(min(x), max(x), 1000)
    x_new2 = np.c_[np.ones((1000, 1)), x_new]
    y_new = x_new2.dot(theta)

    plt.scatter(x, y, c='g')
    plt.plot(x_new, y_new, 'r', linewidth=3)
    plt.show()


if __name__ == '__main__':
    x, y = datasets.make_regression(n_samples=10000, n_features=3, n_targets=1, random_state=11, noise=10)

    theta1, cost1 = gradient_decent(x, y, 100, 100, lr=0.01, type='normal')
    theta2, cost2 = gradient_decent(x, y, 100, 100, lr=0.1, beta=0.9, type='momentum')
    theta3, bias, cost3 = gradient_decent(x, y, 100, 100, lr=0.5, type='adam')
    print(theta1)
    print(theta2)
    print(theta3, bias)

    l=200
    plt.plot(np.arange(l), cost1[:l], '-', c='r', linewidth=1)
    plt.plot(np.arange(l), cost2[:l], '--', c='b', linewidth=1)
    plt.plot(np.arange(l), cost3[:l], '-.', c='g', linewidth=1)
    plt.legend(['normal','momentum', 'adam'],loc='best')
    plt.show()


    # data_plot(x, y, theta)











