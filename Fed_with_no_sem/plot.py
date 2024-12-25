import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成一组示例数据
data = np.random.normal(loc=0, scale=1, size=1000)

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 绘制核密度估计曲线
plt.figure(figsize=(8, 6))
sns.kdeplot(data, color='orange', linewidth=2)
plt.title('Kernel Density Estimation of Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
