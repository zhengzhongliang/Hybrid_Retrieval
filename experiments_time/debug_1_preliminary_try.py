
import time
import numpy as np


n_example = 100

mat_time_list = []
sort_time_list = []

for example in range(n_example):

    # 1: measure time of mat
    matA = np.random.rand(1, 512)
    matB = np.random.rand(512, 100000)

    start_time = time.time()
    _ = np.matmul(matA, matB)
    mat_time_list.append(time.time()-start_time)

    matC = np.random.rand(1, 100000).flatten()
    start_time = time.time()
    _ = np.argsort(matC)

    sort_time_list.append(time.time()-start_time)

print("avg mat time:", sum(mat_time_list)/len(mat_time_list))
print("avg sort time:", sum(sort_time_list)/len(sort_time_list))

