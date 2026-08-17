# -*- coding: utf-8 -*-
"""
分析 Camera Detector 导出的 500x500 网格数据 (波长 1.064 µm, 采样 50 nm)
并重新生成二维热图与三维表面图
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- 读取数据 ----------
pattern = r"C:\Users\Fischer\Desktop\数据\强度\csv格式\csv\Exported_Data_*1.064*.csv"
path = glob.glob(pattern)[0]
print("文件:", path)

data = np.loadtxt(path, delimiter=";", skiprows=7, encoding="utf-8")
print("数据形状:", data.shape)
print("数据范围: min=%.6g  max=%.6g  mean=%.6g" % (data.min(), data.max(), data.mean()))

# ---------- 物理坐标 ----------
# 500 点 x 50 nm = 25 µm, 坐标原点取在中心
n = data.shape[0]              # 500
pitch_um = 0.05                # 50 nm = 0.05 µm
extent_half = n * pitch_um / 2 # 12.5 µm
axis = np.linspace(-extent_half, extent_half, n)  # µm

out_dir = r"C:\Users\Fischer\Desktop\ds2\sessions\分析CSV文件并生成二维三维图"

# ---------- 二维热图 ----------
fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)
im = ax.imshow(
    data,
    extent=[axis[0], axis[-1], axis[0], axis[-1]],
    origin="lower",
    cmap="jet",
    aspect="equal",
)
ax.set_xlabel("x (µm)", fontsize=13)
ax.set_ylabel("y (µm)", fontsize=13)
ax.set_title("Camera Detector @ λ = 1.064 µm — 2D Profile", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Phase (rad)", fontsize=12)
fig.tight_layout()
f2d = out_dir + r"\profile_2D.png"
fig.savefig(f2d, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f2d)

# ---------- 三维表面图 (降采样加速) ----------
step = 3                       # 500 -> ~167 网格
sub = data[::step, ::step]
xs = axis[::step]
ys = axis[::step]
X, Y = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(11, 8.5), dpi=300)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    X, Y, sub,
    cmap="jet",
    rstride=1, cstride=1,
    linewidth=0, antialiased=True,
    edgecolor="none",
)
ax.set_xlabel("x (µm)", fontsize=12, labelpad=10)
ax.set_ylabel("y (µm)", fontsize=12, labelpad=10)
ax.set_zlabel("Phase (rad)", fontsize=12, labelpad=8)
ax.set_title("Camera Detector @ λ = 1.064 µm — 3D Surface", fontsize=14, pad=14)
ax.view_init(elev=32, azim=-128)
cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.08)
cb.set_label("Phase (rad)", fontsize=11)
fig.tight_layout()
f3d = out_dir + r"\profile_3D.png"
fig.savefig(f3d, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f3d)

print("完成")
