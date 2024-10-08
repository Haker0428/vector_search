# vector_search
## 什么是 ANN
近似最近邻（Approximate Nearest Neighbor, ANN） 是一种搜索算法，用于在高维空间中找到与给定查询点最近的点。与精确最近邻搜索不同，ANN 允许在结果中存在一定程度的误差，以换取更快的搜索速度和更低的计算成本。

## Faiss算法使用
### IndexFlat
说明: 这是最基本的索引类型，所有向量都直接存储在内存中，并且查询时使用暴力搜索，计算查询向量与所有数据库向量之间的距离。
**优点**: 精度最高，因为没有近似过程。
**缺点**: 当数据库中的向量数量非常大时，查询速度会很慢，且内存占用高。
**适用场景**: 小规模的数据集或要求极高精度的场景。

```python
index = faiss.IndexFlatL2(d)  # 基于L2距离的暴力搜索
```
### IndexIVF (Inverted File Index)
说明: 通过将向量空间划分成多个簇（使用聚类算法如K-means），只在与查询向量最接近的簇内进行搜索，从而减少计算量。
**优点**: 在保持较高精度的同时大幅提升查询速度。适用于大规模数据。
**缺点**: 在进行搜索之前，需要先进行训练（聚类），并且查询的准确性可能会略低于 IndexFlat。
**适用场景**: 大规模的数据集，尤其是在需要较快的查询速度时。
```python
nlist = 100  # 聚类中心的数量
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
index.train(database_vectors)
index.add(database_vectors)
```


### IndexIVFPQ (Product Quantization)
说明: 结合了倒排文件和产品量化技术。产品量化将向量分成子向量，并对每个子向量进行量化，从而大大减少内存占用和计算量。
优点: 内存效率高，特别适合处理海量数据。
缺点: 需要训练，并且与 IndexFlat 或 IndexIVF 相比，精度可能较低。
适用场景: 超大规模的数据集，内存有限且对速度有较高要求的情况。
```python
m = 8  # 子向量的数量
index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, 8)
index.train(database_vectors)
index.add(database_vectors)
```

### IndexLSH (Locality Sensitive Hashing)
说明: 基于LSH技术，将向量映射到哈希桶中，只对同一哈希桶内的向量进行比较。
优点: 查询速度快，特别适合处理二元向量或稀疏向量。
缺点: 精度相对较低，特别是在高维数据中。
适用场景: 对速度要求极高且对精度要求适中的情况。
```python
index = faiss.IndexLSH(d, 512)  # 512为哈希函数的数量
```
### HNSW (Hierarchical Navigable Small World)
说明: 基于图结构的近似最近邻搜索算法，通过构建多层次的小世界图来快速找到最近邻。
优点: 在大规模数据集上表现非常好，尤其是在高维数据的处理上，查询速度非常快且精度较高。
缺点: 构建索引时间较长，内存占用相对较高。
适用场景: 需要快速、高精度查询的大规模、高维数据集。

```python
index = faiss.IndexHNSWFlat(d, 32)  # 32表示每层的连接数量
```

### IndexBinaryFlat
说明: 适用于二进制向量的暴力搜索索引。二进制向量通常用于描述位串特征，使用Hamming距离计算相似度。
优点: 特别高效地处理二进制数据。
缺点: 仅适用于二进制向量，无法处理浮点数向量。
适用场景: 处理二进制数据，例如用于图像或文档指纹的相似性搜索。
```python
index = faiss.IndexBinaryFlat(d)  # d 是向量的比特长度
```

### IndexFlatIP
说明: 与 IndexFlatL2 类似，但使用内积（dot product）而不是L2距离作为相似度度量。
优点: 适用于需要比较向量之间的内积的场景，例如在某些推荐系统中。
缺点: 与 IndexFlatL2 类似，内存占用高，查询速度随数据量增加而降低。
适用场景: 内积计算是核心操作的场景，例如某些深度学习应用。
```python
index = faiss.IndexFlatIP(d)
```

## 算法
在近似最近邻（ANN）搜索中，要求高精准度和低时延的情况下，一些算法表现优越。以下是基于这些标准的推荐排序，排序顺序从推荐值高到低。

1. HNSW (Hierarchical Navigable Small World Graphs)
优点: 兼具高精度和低时延，尤其在高维数据上表现出色。具有可调节的搜索宽度，可以在速度和精度之间进行平衡。广泛应用于各种场景。
推荐值: ★★★★★
2. IVF-HNSW (Inverted File with HNSW)
优点: 结合了倒排文件（IVF）和 HNSW 的优点，通过分区减少搜索范围，并使用 HNSW 图在分区内进行高效搜索。适合大规模数据集。
推荐值: ★★★★☆
3. IVF-PQ (Inverted File with Product Quantization)
优点: 结合倒排文件和产品量化，IVF-PQ 能够在大规模数据集上实现高效、内存占用低的 ANN 搜索，且可以通过增加 nprobe 提高召回率。
推荐值: ★★★★☆
4. ScaNN (Scalable Nearest Neighbors)
优点: 由 Google 开发，专为高精度和低时延场景设计。通过优化分区和量化策略，ScaNN 能够在大规模数据上实现卓越的性能。
推荐值: ★★★★☆
5. NGT (Neighborhood Graph and Tree)
优点: NGT 的图结构能够有效地实现高精度搜索，结合了图和树结构，适合动态更新的数据集。NGT-qg 变体通过量化进一步优化了速度和内存使用。
推荐值: ★★★★☆
6. Faiss-IVFPQFS (Faiss Inverted File with Fine Quantization and Product Quantization)
优点: Faiss 框架的高级配置，结合细粒度量化和产品量化，提高了精度和查询速度。适用于超大规模数据集。
推荐值: ★★★★☆
7. Annoy (Approximate Nearest Neighbors Oh Yeah)
优点: 使用多个随机树来构建索引，适合静态数据集，查询时延较低，能够在大规模数据集上保持较高的精度。
推荐值: ★★★☆☆
8. FLANN (Fast Library for Approximate Nearest Neighbors)
优点: FLANN 提供了自动调优功能，能够根据数据集选择最合适的算法和参数。适合不同规模的数据集，查询速度快。
推荐值: ★★★☆☆
9. PQ-Search (Product Quantization Search)
优点: 专注于内存效率和搜索速度，适合对内存占用敏感的大规模数据集。虽然在精度上略逊于 IVF-PQ，但在低延迟场景下表现良好。
推荐值: ★★★☆☆
10. LSH (Locality Sensitive Hashing)
优点: 使用哈希函数将相似的向量映射到相同的桶中，适合非常高维的数据和需要极低时延的场景。尽管召回率在某些情况下较低，但时延非常短。
推荐值: ★★★☆☆