import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("Transmission.csv")

# 画图
plt.figure(figsize=(8,6))
plt.scatter(df["Voltage(V)"], df["Transmission(mV)"], color="blue", label="Data")

# 标签 & 标题
plt.xlabel("Voltage(V)")
plt.ylabel("Transmission(mV)")
plt.title("Transmission vs Voltage")
plt.grid(True)
plt.legend()
plt.savefig("figure.svg")   # 矢量图

# 显示图像
plt.show()
