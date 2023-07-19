import numpy as np

def mcmc_binomial(p, n, data, n_iterations):
    """
    使用MCMC方法估计二项分布的参数

    参数：
    p：未知参数p的初始值
    n：数据的样本量
    data：观测到的数据
    n_iterations：MCMC迭代次数

    返回：
    估计得到的参数p的样本
    """
    samples = []
    accepted_samples = 0
    
    for i in range(n_iterations):
        # 生成候选样本
        p_new = np.random.uniform(0, 1)
        
        # 计算接受率
        accept_prob = min(1, np.math.factorial(n) / (np.math.factorial(data) * np.math.factorial(n - data))
                          * pow(p_new, data) * pow(1 - p_new, n - data)
                          / (np.math.factorial(n) / (np.math.factorial(data) * np.math.factorial(n - data))
                             * pow(p, data) * pow(1 - p, n - data)))
        
        # 进行接受-拒绝决策
        if np.random.rand() < accept_prob:
            p = p_new
            accepted_samples += 1
        
        samples.append(p)
    
    acceptance_rate = accepted_samples / n_iterations
    print("接受率：", acceptance_rate)
    
    return samples


# 设置参数和数据
p_true = 0.7
n = 100
data = np.random.binomial(n, p_true)

# 运行MCMC
n_iterations = 10000
samples = mcmc_binomial(0.5, n, data, n_iterations)

# 输出参数估计的均值和标准差
print("参数p的估计均值：", np.mean(samples))
print("参数p的估计标准差：", np.std(samples))