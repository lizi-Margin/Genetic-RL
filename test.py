import numpy as np

# 示例的二维数组
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# 示例的一维索引数组
indices = np.array([0, 1, 2])

# 使用索引数组获取第二维度的数据
output_array = arr[indices]

# 输出结果
print("输出的一维数组:", output_array)

