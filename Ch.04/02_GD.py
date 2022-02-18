# %%
import numpy as np
import matplotlib.pyplot as plt
from torch import gradient

# 선형 회귀 모델의 예측


# 1. 임의 데이터 생성
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# print(X)

# 2. 데이터 분포도 확인
plt.plot(X, y, "b.") # 'b.' 이 자리에 아무것도 없으면 선
plt.xlabel("x_1", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15]) # [xmin, xmax, ymin, ymax]
# save_fig("generated_data_plot")
plt.show()

# %%

# 3. 정규방적식을 통해 예측하기 (p.160)
X_b = np.c_[np.ones((100, 1)), X]  # 모든 샘플에 x0 = 1을 추가합니다. (100, 2) 

# X_b.T.dot(X_b)
# X_b = (100,2)
# X_b.T = (2,100)
# X_b.T.dot(X_b) = (2, 100) dot (100, 2) = (2, 2)

# (2,2) dot (2,100) = (2, 100)
# (2,100) dot (100, 1) = (2, 1)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # 정규방정식
theta_best


# %%

# 4. 예측 결과 확인
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # 모든 샘플에 x0 = 1을 추가합니다.
y_predict = X_new_b.dot(theta_best)
print('y_predict = ',y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()


# %%

# 5. 경사하강법

eta = 0.1 # 학습률
n_iterations = 1000
m = 100

theta = np. random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
print(X_new_b.dot(theta))

# %%

# 6. 학습률에 따른 경사하강법 변화 확인

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # 정규방정식 활용
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)


plt.show()
# %%

# 7. 확률적 경사하강법
n_epochs = 50
t0, t1 = 5, 50 # 학습 스케줄 하이퍼파라미터

theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # 무작위 초기화

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)   
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) # 정규방정식 활용
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
    
print(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()        



# %%
  


# %%
