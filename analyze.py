import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
######

# -------------------
# 1. 读入数据
# -------------------
fname = "Result_780PM.csv"

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
#x_fit = np.concatenate([x[:20], x[-10:]])
#y_fit = np.concatenate([y[:60], y[-10:]])
x_fit=x
y_fit=y
popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=100000)
perr = np.sqrt(np.diag(pcov))
I0, x0, sigma, C = popt

# -------------------
# 5. 结果 (只对拟合点计算R²)
# -------------------
yfit_all = gaussian(x, *popt)             # 全部点的拟合值 (用于画图)
yfit_fit = gaussian(x_fit, *popt)         # 仅拟合点的拟合值 (用于R²)

ss_res = np.sum((y_fit - yfit_fit)**2)
ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
r2 = 1 - ss_res/ss_tot

w = 2 * sigma * 2.54
w_err = 2 * perr[2] * 2.54
diam = 2 * w 
diam_err = 2 * w_err 

print("Gaussian fit parameters:")
print(f"I0    = {I0:.3f} ± {perr[0]:.3f}")
print(f"x0    = {x0:.3f} ± {perr[1]:.3f}")
print(f"sigma = {sigma:.3f} ± {perr[2]:.3f}")
print(f"C     = {C:.3f} ± {perr[3]:.3f}")
print(f"R^2 (fit points only) = {r2:.4f}")
print(f"Beam radius w (1/e^2) = {w:.3f} ± {w_err:.3f}")
print(f"Beam diameter = {diam:.3f} ± {diam_err:.3f}")


# -------------------
# 6. 绘图
# -------------------
xfit = np.linspace(min(x), max(x), 600)

plt.figure(figsize=(7,5))

# 所有数据点：黑色
plt.scatter(x, y, color='black', s=30, label='all data')

# 拟合用的数据点：蓝色
plt.scatter(x_fit, y_fit, color='blue', s=50, label='fit data', zorder=3)

# 高斯拟合曲线
plt.plot(xfit, gaussian(xfit, *popt), color='tab:orange', lw=2, 
         label=f'Gaussian fit (R²={r2:.3f})')

# 坐标轴标签
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 在图中添加光束直径信息
plt.text(0.05, 0.95, 
         f"Beam diameter = {diam:.2f} ± {diam_err:.2f}", 
         transform=plt.gca().transAxes, 
         fontsize=10, 
         verticalalignment="top")

plt.tight_layout()
plt.savefig("gaussian_fit.svg") 
plt.show()
