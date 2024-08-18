#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// 定义向量的维度和子空间数量
#define DIM 128
#define M 8  // 子向量的数量
#define K 256  // 每个子空间的质心数量
#define SUBDIM (DIM / M)  // 每个子向量的维度

// 计算两个向量之间的欧氏距离
float euclidean_distance(const float *a, const float *b, int d) {
    float dist = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

// 简单的k-means算法，用于训练质心
void kmeans(float *data, int n, int d, int k, float *centroids, int *assignments) {
    // 随机初始化质心
    for (int i = 0; i < k * d; i++) {
        centroids[i] = data[rand() % (n * d)];
    }

    int changed;
    do {
        changed = 0;

        // 更新分配
        for (int i = 0; i < n; i++) {
            int best_centroid = 0;
            float best_dist = FLT_MAX;
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(&data[i * d], &centroids[j * d], d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_centroid = j;
                }
            }
            if (assignments[i] != best_centroid) {
                assignments[i] = best_centroid;
                changed++;
            }
        }

        // 更新质心
        for (int j = 0; j < k; j++) {
            int count = 0;
            for (int l = 0; l < d; l++) {
                centroids[j * d + l] = 0.0f;
            }
            for (int i = 0; i < n; i++) {
                if (assignments[i] == j) {
                    for (int l = 0; l < d; l++) {
                        centroids[j * d + l] += data[i * d + l];
                    }
                    count++;
                }
            }
            if (count > 0) {
                for (int l = 0; l < d; l++) {
                    centroids[j * d + l] /= count;
                }
            }
        }
    } while (changed > 0);
}

// 对输入向量进行PQ编码
void pq_encode(float *vec, float *centroids, int *codes) {
    for (int m = 0; m < M; m++) {
        int best_centroid = 0;
        float best_dist = FLT_MAX;
        for (int j = 0; j < K; j++) {
            float dist = euclidean_distance(&vec[m * SUBDIM], &centroids[m * K * SUBDIM + j * SUBDIM], SUBDIM);
            if (dist < best_dist) {
                best_dist = dist;
                best_centroid = j;
            }
        }
        codes[m] = best_centroid;
    }
}

// 解码PQ编码得到近似向量
void pq_decode(int *codes, float *centroids, float *approx_vec) {
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < SUBDIM; i++) {
            approx_vec[m * SUBDIM + i] = centroids[m * K * SUBDIM + codes[m] * SUBDIM + i];
        }
    }
}

int main() {
    int n = 1000;  // 数据集的向量数量
    float data[n * DIM];  // 原始数据集
    float centroids[M * K * SUBDIM];  // 每个子空间的质心
    int assignments[n * M];  // 每个子空间的聚类结果
    int codes[M];  // PQ编码
    float approx_vec[DIM];  // 解码后的近似向量

    // 随机生成数据集
    for (int i = 0; i < n * DIM; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }

    // 对每个子空间进行k-means训练
    for (int m = 0; m < M; m++) {
        kmeans(&data[m * SUBDIM], n, SUBDIM, K, &centroids[m * K * SUBDIM], &assignments[m * n]);
    }

    // 对一个向量进行PQ编码
    pq_encode(data, centroids, codes);

    // 将编码结果解码为近似向量
    pq_decode(codes, centroids, approx_vec);

    // 输出原始向量和解码后的近似向量
    printf("Original vector: ");
    for (int i = 0; i < DIM; i++) {
        printf("%.3f ", data[i]);
    }
    printf("\nApproximated vector: ");
    for (int i = 0; i < DIM; i++) {
        printf("%.3f ", approx_vec[i]);
    }
    printf("\n");

    return 0;
}
