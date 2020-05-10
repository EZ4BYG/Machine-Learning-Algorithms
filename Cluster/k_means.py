import numpy as np
import matplotlib.pyplot as plt

# 数据：我想分4组数据
x1 = np.random.uniform(0, 10, 20)
y1 = np.random.uniform(0, 10, 20)

x2 = np.random.uniform(10, 20, 20)
y2 = np.random.uniform(0, 10, 20)

x3 = np.random.uniform(10, 20, 20)
y3 = np.random.uniform(10, 20, 20)

x4 = np.random.uniform(0, 10, 20)
y4 = np.random.uniform(10, 20, 20)

x = np.concatenate( (x1,x2,x3,x4) )
y = np.concatenate( (y1,y2,y3,y4) )

plt.scatter(x,y)
plt.show()

# 数据初始化：
k = 4  # 分几份，自己决定
data_total = len(x)
# 一开始的k个随机点坐标：把x、y坐标分开记录
range_x = max(x) - min(x)
range_y = max(y) - min(y)
ax = []
ay = []
for i in range(k):
    atmp_x = np.random.uniform(0.001,1) * range_x  # 初始随机值对最后的划分：影响很大！ —— 需要多试几次！
    atmp_y = np.random.uniform(0.001,1) * range_y
    ax.append( atmp_x )
    ay.append( atmp_y )


# 1. 数组分组函数：每一次，对每个点归类：0, 1, 2, ..., k-1
def min_distance_index(each_ax, each_ay):
    clusterx = []
    clustery = []
    distance = np.zeros((k, data_total))

    # 记录“每个点”，到每个大哥点each_a的距离
    for each_a in range(k):
        x_a = np.power(x - each_ax[each_a], 2)  # X
        y_a = np.power(y - each_ay[each_a], 2)  # Y
        distance[each_a] = np.sqrt(x_a + y_a)  # 每行元素：都是数据点到i大哥点的“距离值”

    min_distance_index = np.argmin(distance, axis=0)  # 列最小值索引：每个点距离哪个“大哥点”最近
    for i in range(k):
        index = np.where(min_distance_index == i)  # 归类
        clusterx.append(x[index])
        clustery.append(y[index])

    return clusterx, clustery  # 数据已分组：每个变量里4个元素/分组，每个元素里又是数组


# 2. 每组质心求取函数：每组x、y坐标的均值
def mean_center(clusterx, clustery):
    # 记录每组质心点x、y坐标：
    newa_x = []
    newa_y = []
    # 对“每个组i”进行质心坐标x、y循环求取：
    for i in range(k):
        axtmp = np.mean(clusterx[i])
        aytmp = np.mean(clustery[i])
        newa_x.append(axtmp)
        newa_y.append(aytmp)

    return np.array(newa_x), np.array(newa_y)

# 主体部分：
for i in range(100):
    clusterx, clustery = min_distance_index(ax, ay)  # 分组后数据
    axtmp, aytmp = mean_center(clusterx, clustery)   # 新质心
    # 计算误差：axtmp与ax
    dx = np.power( axtmp - ax, 2 )
    dy = np.power( aytmp - ay, 2 )
    error = np.sum( np.sqrt( dx + dy ) )  # 一定会收敛！
    print('当前误差：', error)
    if error < 0.001:
        print('循环次数：', i, '精度要求已达标，提前结束！')
        break
    else:
        ax = axtmp
        ay = aytmp

# 画图显示：
for i in range(k):
    plt.scatter(clusterx[i], clustery[i])
plt.show()