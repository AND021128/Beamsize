import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------
# 1. 读入数据
# -------------------
fname = "Result_630PM.csv"

# 如果 CSV 第一行有表头，就用 header=0；如果没有，就改成 header=None
df = pd.read_csv(fname, header=0)  
x = df.iloc[:,0].values.astype(float)
y = df.iloc[:,1].values.astype(float)

# -------------------
# 2. 定义高斯函数
# -------------------
def gaussian(x, I0, x0, sigma, C):
    return I0 * np.exp(- (x - x0)**2 / (2 * sigma**2)) + C

# -------------------
# 3. 初始猜测参数
# -------------------
I0_guess = max(y) - min(y)
x0_guess = x[np.argmax(y)]
sigma_guess = (max(x)-min(x))/6
C_guess = min(y)
p0 = [I0_guess, x0_guess, sigma_guess, C_guess]

# -------------------
# 4. 拟合
# -------------------
x_fit = np.concatenate([x[:14], x[-13:]])
y_fit = np.concatenate([y[:14], y[-13:]])
popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=100000)
perr = np.sqrt(np.diag(pcov))
I0, x0, sigma, C = popt

# -------------------
# 5. 结果
# -------------------
yfit = gaussian(x, *popt)
ss_res = np.sum((y - yfit)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot

w = 2 * sigma
w_err = 2 * perr[2]
diam = 2 * w
diam_err = 2 * w_err

print("Gaussian fit parameters:")
print(f"I0    = {I0:.3f} ± {perr[0]:.3f}")
print(f"x0    = {x0:.3f} ± {perr[1]:.3f}")
print(f"sigma = {sigma:.3f} ± {perr[2]:.3f}")
print(f"C     = {C:.3f} ± {perr[3]:.3f}")
print(f"R^2   = {r2:.4f}")
print(f"Beam radius w (1/e^2) = {w:.3f} ± {w_err:.3f}")
print(f"Beam diameter = {diam:.3f} ± {diam_err:.3f}")

# -------------------
# 6. 绘图
# -------------------
xfit = np.linspace(min(x), max(x), 600)
plt.figure(figsize=(7,5))
plt.scatter(x, y, color='black', s=30, label='data')
plt.plot(xfit, gaussian(xfit, *popt), color='tab:blue', lw=2, label=f'Gaussian fit (R²={r2:.3f})')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("gaussian_fit.svg") 
plt.show()
