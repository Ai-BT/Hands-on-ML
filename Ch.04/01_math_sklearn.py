# %%
import numpy as np
import matplotlib.pyplot as plt

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

# 5. 예측선 그리기
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("x_1", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()
# %%

# 6. 사이킷런에서 선형 회구를 수행
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

# %%
