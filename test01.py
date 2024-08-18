import numpy as np
import faiss
import time

class Solution:
    def __init__(self):
        self.index = None
        self.base = None
        self.k = 10

    def build(self, base: np.ndarray[[float, float]]):
        self.base = base
        d = base.shape[1]
        # 创建一个基于L2距离的索引
        self.index = faiss.IndexFlatL2(d)
        # 添加向量到索引中
        self.index.add(base)
        print("Index built with {} vectors".format(base.shape[0]))

    def search(self, query: np.ndarray) -> np.ndarray:
        # 查询，k=10表示查找最接近的10个向量
        distances, indices = self.index.search(query.reshape(1, -1), self.k)
        print("Query results:", indices)
        return indices


# 测试代码
if __name__ == "__main__":
    # 创建一个 Solution 实例
    sol = Solution()

    # 创建一个随机的 base 和 query 数据
    n = 100000  # 底库向量数量
    m = 5  # 查询向量数量
    d = 256  # 向量维度

    # 固定随机数种子以获得可重复的结果
    np.random.seed(42)
    base = np.random.random((n, d)).astype('float32')
    query = np.random.random((m, d)).astype('float32')

    # 构建索引
    sol.build(base)
    # 记录开始时间
    start_time = time.time()
    # 执行查询
    results = sol.search(query[0])
    # 记录结束时间
    end_time = time.time()

    # 计算并输出时间
    elapsed_time = end_time - start_time
    print(f"Search completed in {elapsed_time:.6f} seconds")
    # 输出查询结果
    print("Top 10 nearest vectors for each query:\n", results)
