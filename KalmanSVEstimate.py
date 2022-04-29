# encoding=utf-8
# By Liuyiwen
# 765305261@qq.com

"""
python3做匀加速小车位置估计的Kalman滤波模拟:
在预测值和传感器测量值之间找到最优解, 是一个最优化后验概率(MAP)的过程。
    将两个高斯概率分布相乘得到重叠部分: 仍然是一个高斯概率分布(融合高斯分布)
    [高斯融合]的结果：
        相当于用把估计的均值取了预测和观测之间的(权衡值)，
        且把估计的方差减小了，这就是[数据融合]的主要目的。
(1). 系统的状态变量总共2个: 位移S和速度v,且假设这两个系统状态变量不相关
(2). 系统建模——状态转移方程:
            [St, Vt]' = [[1, Δt], *  [S(t-1), V(t-1)]' + [1/2*(Δt)^2, Δt]' * a(加速度)
                         [0, 1]]
     系统建模——观测方程:
            [St', Vt'] = [[1, 0], * [St, Vt]' + [ΔSt, ΔVt](误差或噪声)
                           0, 1] 
            测量方式为直接获得位置坐标而无需公式转换，所以观测矩阵H中位移为1,速度为1
"""

import matplotlib.pyplot as plt
import numpy as np

dt = 0.1  # 采样间隔时间(单位s)
t = np.linspace(0, 10, 101, endpoint=True)  # 时间序列
# print('t:', t)
N = len(t)
# print(N)
size = (N, 2)
print(size)
g = 10  # 重力加速度, 模拟自由落体运动位置估计
shift_real = 0.5 * g * t * t  # 真实位移值
velocity_real = g * t  # 真是速率值
# print('x: ', x)
shift_noise = np.random.normal(0, 13.5, size=N)  # 位移高斯白噪声
velocity_noise = np.random.normal(0, 9.9, size=N)  # 速率高斯白噪声
# size_1 = int(round(0.375*N))
# noise_1 = np.random.normal(0, 9.0, size=size_1)
# noise_2 = np.random.normal(0, 11.0, size=N-size_1)
# noise = np.append(noise_1, noise_2) # numpy数组拼接

S_measure = shift_real + shift_noise  # 加入高斯白噪声的位移测量值
V_measure = velocity_real + velocity_noise  # 加入高斯暴躁生的速率测量值

## -----协方差矩阵: 描述变量的相关性
# 过程噪声/扰动(预测噪声)的协方差矩阵:
# 协方差矩阵, 非对角线为0: 两个状态变量独立分布: 位置和速度是不相关的
# 协方差矩阵, 非对角线不为0: 两个状态变量非独立分布: 位置和速度是线性相关的
# 位移s的过程噪声方差为0
Q = np.array([[0.2, 0.1],
              [0.1, 0.2]])
print("Q:\n", Q)

# 系统测量噪声(状态转移噪声/扰动)为常量
# 例如：传感器噪声, 用协方差矩阵R表示
# 测量噪声R增大,动态响应变慢,收敛稳定性变好
R = np.array([[3.0, 0.5],  # R该如何设置初值?
              [0.5, 3.0]])  # R太小不会收敛
print("R:\n", R)

# x系统建模：系统状态转移矩阵A(2*2)
A = np.array([[1, dt], [0, 1]])
print("A:\n", A)

# 控制矩阵B(2*1)
B = np.array([0.5 * dt * dt, dt])
print("\n:", B)

# 外部控制量: 控制输入向量, eg: 重力加速度

# 测量转换矩阵H: 测量值当然是由系统状态变量映射出来的,
# eg: 量纲/尺度转换, 将状态变量值转换成传感器读数值
# 系统状态变量到实际测量值的转换矩阵: z = h*x+v(测量噪声)
# 系统状态变量中,既测量位移,也测量速度，此时相当于单位矩阵
H = np.array([[1, 0],
              [0, 1]])

# 系统初始化
n = Q.shape
# print('n:', n)

m = R.shape
# print('m:', m)

x_hat = np.zeros(size)  # x的后验估计值
print("x_hat:\n", x_hat)

# 后验估计误差协方差矩阵(每次迭代最后需更新)
P = np.eye(2)
print("P:\n", P)

# x的先验估计值(上一次估计值, 每次迭代开始需更新)
x_hat_minus = np.zeros(size)

# 先验估计误差协方差矩阵(每次迭代开始需更新)
P_minus = np.zeros(n)
# print('P_minus init:', P_minus)

# Kalman增益矩阵(2*2)
K = np.zeros((n[0], m[0]))
# print('K:', K)

I = np.eye(n[0], n[1])  # 单位矩阵
# print('I:', I)

# Kalman迭代过程
for i in range(1, N):
    ## --------- predict
    # t-1 到 t时刻的状态预测，得到前验概率
    # (1).状态转移方程
    # x_hat_minus[i] = A.dot(x_hat[i - 1]) + B * g
    x_hat_minus[i] = np.dot(A, x_hat[i - 1]) + B * g

    # (2).误差转移方程
    # P_minus = A.dot(P).dot(A.T) + Q  
    P_minus = np.linalg.multi_dot((A, P, A.T)) + Q

    ## ---------- update
    # 根据观察量对预测状态进行修正，得到后验概率，也就是最优值
    # (3).Kalman增益
    # K = P_minus.dot(H.T).dot(np.linalg.inv(H.dot(P_minus).dot(H.T) + R))
    K = P_minus.dot(H.T).dot(np.linalg.inv(np.linalg.multi_dot((H, P_minus, H.T)) + R))
    print('\n--Round %d K:\n' % i, K)

    # (4).状态修正方程
    z = np.array([S_measure[i], V_measure[i]])  # 测量值, eg: 传感器数据
    x_hat[i] = x_hat_minus[i] + K.dot(z - H.dot(x_hat_minus[i]))

    # (5).误差修正方程
    print(K.dot(H))
    P = (I - K.dot(H)).dot(P_minus)
    # P = P_minus - K.dot(H).dot(P_minus)
    # P = P_minus - np.linalg.multi_dot((K, H, P_minus))
    print('--Round %d P:\n' % i, P)

# 取位移和速度的估计值
S_estimate = [s for (s, v) in x_hat]
V_estimate = [v for (s, v) in x_hat]

# Kalman迭代过程
plt.figure()
plt.plot(S_measure, 'r+', label='measured shift')  # 测量位移值
plt.plot(V_measure, 'm+', label='measured velocity')  # 测量速率值
plt.plot(S_estimate, 'b-', label='estimated shift')  # 估计位移值
plt.plot(V_estimate, 'b-', label='estimated velocity')  # 估计速率值
plt.plot(shift_real, 'y-', label='real shift')  # 真实位移值
plt.plot(velocity_real, 'y-', label='real velocity')  # 真实速率值
plt.legend()
plt.title('Kalman filter')
plt.xlabel('Iteration')
plt.ylabel('Shift & Velocity')
plt.tight_layout()
plt.show()

# 下一步考虑track cross坐标定位的Kalman建模：从单个坐标到81个坐标
# 下一步考虑Intensity建模
# 下一步考虑扩展卡尔曼滤波算法(EKF)对非线性系统建模
# ref:
# https://blog.csdn.net/u010720661/article/details/63253509
