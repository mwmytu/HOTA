'''
泊松分布
'''
import numpy as np
import math
import matplotlib.pyplot as plt


# 定义泊松分布的概率质量函数
def poisson_pmf(k, lambda_):
    return (lambda_ ** k) * np.exp(-lambda_) / math.factorial(k)


# 定义混合泊松分布的概率密度函数
def mixed_poisson_pmf(k, lambdas, weights):
    result = 0
    for i in range(len(lambdas)):
        result += weights[i] * poisson_pmf(k, lambdas[i])
    return result


# 定义参数和权重
lambdas = [8, 12, 17]
weights = [0.3, 0.3, 0.3]

# 计算概率密度函数值
k_values = np.arange(0, 24)
pmf_values = [mixed_poisson_pmf(k, lambdas, weights) for k in k_values]

# 绘制图像
plt.bar(k_values, pmf_values, width=0.1)
plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Mixed Poisson Distribution')
plt.show()
