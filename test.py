import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
data = np.random.rand(100, 100)

# 绘制热图
plt.imshow(data, cmap='viridis')
plt.colorbar()  # 显示颜色条
plt.title("Matplotlib 热图")
plt.show()
