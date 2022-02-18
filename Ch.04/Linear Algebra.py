# 단위행렬 (Unit matrix): np.eye(n)
# 대각행렬 (Diagonal matrix): np.diag(x)
# 내적 (Dot product, Inner product): np.dot(a, b)
# 대각합 (Trace): np.trace(x)
# 행렬식 (Matrix Determinant): np.linalg.det(x)
# 역행렬 (Inverse of a matrix): np.linalg.inv(x)
# 고유값 (Eigenvalue), 고유벡터 (Eigenvector): w, v = np.linalg.eig(x)
# 특이값 분해 (Singular Value Decomposition): u, s, vh = np.linalg.svd(A)
# 연립방정식 해 풀기 (Solve a linear matrix equation): np.linalg.solve(a, b)
# 최소자승 해 풀기 (Compute the Least-squares solution): m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# https://rfriend.tistory.com/380 참고

# %%
import numpy as np

# 1. 단위행렬 (항등행렬) eye
unit_mat_4 = np.eye(4)
print(unit_mat_4)
# %%

# 2. 대각행렬 diag
# 대각성분 이외의 모든 성분이 모두 '0'인 n차 정방행렬

x = np.arange(9).reshape(3,3)
print(x)

print('a = ',np.diag(x))
print('b = ',np.diag(np.diag(x)))


# %%

# 3. 내적 Dot

# 원소간 곱
a = np.arange(4).reshape(2,2)
print(a)
print(a*a)

# 내적 곱
b = np.dot(a,a)
print(b)

# or
c = a.dot(a)
print(c)


# %%

# 4. 역행렬 linalg
# A^-1 = 1/ad-bc([[d, -b, -c, a]])
# ad-bc =! 0 이면 역행렬 존재 x
# ad-bc = 0 이면 역행렬 있음
# https://mathbang.net/567 참고

a = np.arange(4).reshape(2,2)
print(a)

a_inv = np.linalg.inv(a)
print(a_inv)

# %%

# 5. concatenate
# c_ = concatenate
# 연쇄시키다

X = 2 * np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
print(X_b)
print(X_b.shape)

# %%

# 0 - 1 의 균일분포 표준정규분포 난수를 생성
a = np.random.rand(3,2)
print(a)

# 평균 0, 표준편차 1의 가우시안 표준정규분포 난수를 생성
b = np.random.randn(3,2)
print(b)

# %%
