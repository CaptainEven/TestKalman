# TestKalman

通过python3改写匀加速小车位置估计的Kalman滤波模拟
(1). 系统的状态变量总共2个: 位移S和速度v,且假设这两个系统状态变量不相关
(2). 系统建模——状态转移方程:
            [St, Vt]' = [[1, Δt], *  [S(t-1), V(t-1)]' + [1/2*(Δt)^2, Δt]' * a(加速度)
                         [0, 1]]
     系统建模——观测方程:
            [St', Vt'] = [[1, 0], * [St, Vt]' + [ΔSt, ΔVt](误差或噪声)
                           0, 1] 
            测量方式为直接获得位置坐标而无需公式转换，所以观测矩阵H中位移为1,速度为1

# Demo figure
![image](https://github.com/CaptainEven/MCMOT-ByteTrack/blob/master/test_13.gif)
