# -*- coding: utf-8 -*-
"""
DOE 相位设计: 50 mm 工作距离再现目标图片 (罗小黑)
- DOE 1.5x1.5 mm, 目标图 40x40 mm, 高斯光束入射, 自由空间 (Fresnel) 传播
- 信号窗口 IFTA (Gerchberg-Saxton 改进型), 多次随机重启取最优
输出: 相位图 / 重建图 / 对比图 / 收敛曲线 / metrics + 相位数据 npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["mathtext.fontset"] = "cm"

TASK_DIR = r"C:\Users\Fischer\Desktop\ds2\auto_sender\202608171855"
IMG_PATH = r"C:\Users\Fischer\Desktop\图片\罗小黑战记\罗小黑.jpeg"

# ---------------- 系统参数 ----------------
LAM = 1.064e-6          # 波长 (沿用本批次数据波长, 报告中注明)
Z = 50e-3               # 工作距离 50 mm
L1 = 1.5e-3             # DOE 尺寸 1.5 mm
N = 1536                # DOE 采样点数 (dx1 ~ 0.98 µm)
W0 = 0.75e-3            # 入射高斯 1/e^2 半径 (假设值)
IMG_SIZE = 40e-3        # 目标图尺寸 40 mm
N_RESTART, N_ITER = 3, 120
CDT = np.complex64          # 单精度复数加速

dx1 = L1 / N
dx2 = LAM * Z / L1                      # 输出采样 35.47 µm
L2 = N * dx2                            # 输出窗口 ~54.5 mm
n_sig = int(round(IMG_SIZE / dx2))      # 信号窗口像素数 (~1128)
c0 = (N - n_sig) // 2

x1 = (np.arange(N) - N // 2) * dx1
X1, Y1 = np.meshgrid(x1, x1)
R1 = np.sqrt(X1**2 + Y1**2)
x2 = (np.arange(N) - N // 2) * dx2

print("dx1 = %.3f µm, dx2 = %.2f µm, L2 = %.1f mm, 信号窗口 = %d px"
      % (dx1 * 1e6, dx2 * 1e6, L2 * 1e3, n_sig))
print("最大衍射角(图边缘 20 mm) = %.1f deg" % np.degrees(np.arctan(20 / 50)))
print("Nyquist 允许最大坐标 = %.1f mm" % (LAM * Z / (2 * dx1) * 1e3))

# ---------------- 目标图像 ----------------
img = Image.open(IMG_PATH).convert("L").resize((n_sig, n_sig), Image.LANCZOS)
T = np.asarray(img, dtype=np.float64) / 255.0     # 目标强度 [0,1]

A_t = np.zeros((N, N))
A_t[c0:c0 + n_sig, c0:c0 + n_sig] = np.sqrt(T)    # 目标振幅
SW = np.zeros((N, N), dtype=bool)
SW[c0:c0 + n_sig, c0:c0 + n_sig] = True
At_energy = (A_t[SW] ** 2).sum()

# ---------------- 入射高斯场 ----------------
A_in = np.exp(-(R1 / W0) ** 2)

# Fresnel 输入面二次相位 (输出面二次相位在往复中抵消, 省略)
Q_in = np.exp(1j * np.pi * R1**2 / (LAM * Z))

# 预移位约定: 循环内不做 fftshift, 全部数组转到 FFT 自然顺序
Q_s = np.fft.ifftshift(Q_in).astype(CDT)
Ain_s = np.fft.ifftshift(A_in).astype(np.float32)
At_s = np.fft.ifftshift(A_t).astype(np.float32)
SW_s = np.fft.ifftshift(SW)
At_energy = float((At_s[SW_s] ** 2).sum())
t_vec = T.ravel()


def fwd(E):
    return np.fft.fft2(E * Q_s)


def inv(U):
    return np.conj(Q_s) * np.fft.ifft2(U)


def rmse_of(amp):
    """amp 为 FFT 自然布局; 转到中心布局后再与目标比较"""
    ac = np.fft.fftshift(amp)
    Isw = (ac[SW] ** 2).astype(np.float64)
    g = (Isw * t_vec).sum() / max((Isw**2).sum(), 1e-30)
    return float(np.sqrt(np.mean((g * Isw - t_vec) ** 2))), float(g)


# ---------------- IFTA (信号窗口约束 + MRAF 混合) ----------------
M_MIX = 0.8             # MRAF 混合因子: 目标振幅占比


def ifta(seed):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (N, N)).astype(np.float32)
    E = (Ain_s * np.exp(1j * phase)).astype(CDT)
    hist = []
    for it in range(N_ITER):
        U = fwd(E)
        amp = np.abs(U)
        r, _ = rmse_of(amp)
        hist.append(r)
        # 信号窗口内能量归一: 目标振幅缩放到当前窗口能量
        e_sw = float((amp[SW_s] ** 2).sum())
        s = np.sqrt(e_sw / At_energy)
        # MRAF: m*目标 + (1-m)*当前振幅, 相位保留
        mixed = (M_MIX * s * At_s[SW_s]
                 + (1.0 - M_MIX) * amp[SW_s])
        U_new = U.copy()
        U_new[SW_s] = (mixed
                       * np.exp(1j * np.angle(U[SW_s]))).astype(CDT)
        E = (Ain_s * np.exp(1j * np.angle(inv(U_new)))).astype(CDT)
    return np.angle(E), hist


print("IFTA 设计中 (%d 次重启 x %d 次迭代)..." % (N_RESTART, N_ITER))
best = None
for sd in range(N_RESTART):
    ph, hist = ifta(sd)
    r_final = hist[-1]
    E_n = (Ain_s * np.exp(1j * ph)).astype(CDT)
    amp = np.abs(fwd(E_n))
    eta = float((amp[SW_s] ** 2).sum() / (amp**2).sum())
    print("  重启 %d: RMSE = %.5f, 效率 = %.4f" % (sd, r_final, eta),
          flush=True)
    if best is None or r_final < best[0]:
        best = (r_final, ph, hist, eta)

rmse_c, phase_nat, hist_c, eta_c = best
phase_doe = np.fft.fftshift(phase_nat)      # 转回物理(中心)布局
print("最优: RMSE = %.5f, 衍射效率 = %.4f" % (rmse_c, eta_c))

# ---------------- 重建仿真 (连续 + 256 级量化) ----------------
def reconstruct(ph):
    """物理布局相位 -> Fresnel 传播 -> 光强 (中心布局)"""
    E = A_in * np.exp(1j * ph)
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E * Q_in)))
    return np.abs(U) ** 2


I_rec = reconstruct(phase_doe)
levels = 256
ph_q = np.round(phase_doe / (2 * np.pi / levels)) * (2 * np.pi / levels)
I_q = reconstruct(ph_q)

eta_q = float(I_q[SW].sum() / I_q.sum())
Isw_q = I_q[SW]
g_q = (Isw_q * t_vec).sum() / max((Isw_q**2).sum(), 1e-30)
rmse_q = float(np.sqrt(np.mean((g_q * Isw_q - t_vec) ** 2)))
print("256 级量化: RMSE = %.5f, 效率 = %.4f" % (rmse_q, eta_q))

# 连续相位的 RMSE (中心布局)
Isw_c = I_rec[SW]
g_c = (Isw_c * t_vec).sum() / max((Isw_c**2).sum(), 1e-30)
rmse_cc = float(np.sqrt(np.mean((g_c * Isw_c - t_vec) ** 2)))
print("连续相位 (中心布局复核): RMSE = %.5f, 效率 = %.4f"
      % (rmse_cc, eta_c))

# 显示用强度: 按最小二乘增益缩放 (物理重建为 180° 旋转像)
I_show = np.clip(g_c * I_rec, 0, None)
mc = np.abs(x2) <= 21e-3
ax2c = x2[mc] * 1e3
ext = [ax2c[0], ax2c[-1]] * 2
I_rec_c = I_show[np.ix_(mc, mc)]
I_rec_c /= I_rec_c.max()
T_rot = np.rot90(T, 2)          # 旋转目标以便对比

# ---------------- 图 1: 目标图 ----------------
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300)
ax.imshow(T, extent=[-20, 20, -20, 20], origin="lower",
          cmap="gray", vmin=0, vmax=1)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title("Target image (40 mm × 40 mm)", fontsize=13)
fig.tight_layout()
fig.savefig(TASK_DIR + r"\doe_target.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存: doe_target.png")

# ---------------- 图 2: DOE 相位 (0-2pi, gray, LaTeX colorbar) ----------------
fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
im = ax.imshow(phase_doe, extent=[x1[0] * 1e3, x1[-1] * 1e3] * 2,
               origin="lower", cmap="gray", vmin=0, vmax=2 * np.pi,
               aspect="equal")
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title("Designed DOE phase (1.5 mm × 1.5 mm)", fontsize=13)
cb = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0, 2 * np.pi])
cb.set_ticklabels([r"$0$", r"$2\pi$"], fontsize=13)
cb.set_label("Phase (rad)", fontsize=12)
fig.tight_layout()
fig.savefig(TASK_DIR + r"\doe_phase.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存: doe_phase.png")

# ---------------- 图 3: 重建光强 ----------------
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300)
ax.imshow(I_rec_c, extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title("Simulated reconstruction @ z = 50 mm", fontsize=13)
fig.tight_layout()
fig.savefig(TASK_DIR + r"\doe_reconstruction.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("已保存: doe_reconstruction.png")

# ---------------- 图 4: 目标 vs 重建 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=300)
axes[0].imshow(T_rot, extent=[-20, 20, -20, 20], origin="lower",
               cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Target (rotated 180°)", fontsize=13)
axes[0].set_xlabel("x (mm)")
axes[0].set_ylabel("y (mm)")
axes[1].imshow(I_rec_c, extent=ext, origin="lower", cmap="gray",
               vmin=0, vmax=1)
axes[1].set_title("DOE reconstruction", fontsize=13)
axes[1].set_xlabel("x (mm)")
for a in axes:
    a.set_xlim(-21, 21)
    a.set_ylim(-21, 21)
fig.suptitle("Target vs simulated reconstruction @ 50 mm "
             "(propagation inverts image 180°)", fontsize=13)
fig.tight_layout()
fig.savefig(TASK_DIR + r"\doe_compare.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存: doe_compare.png")

# ---------------- 图 5: 收敛曲线 ----------------
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)
ax.plot(np.arange(1, N_ITER + 1), hist_c, color="#c0362c", lw=1.5)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("RMSE (signal window)", fontsize=12)
ax.set_title("IFTA convergence (best restart)", fontsize=13)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(TASK_DIR + r"\doe_convergence.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("已保存: doe_convergence.png")

# ---------------- 保存数据与指标 ----------------
np.savez_compressed(
    TASK_DIR + r"\doe_phase_data.npz",
    phase=phase_doe.astype(np.float32),
    phase_256level=ph_q.astype(np.float32),
    dx1_um=dx1 * 1e6, lam_um=LAM * 1e6, z_mm=Z * 1e3, w0_mm=W0 * 1e3,
)
np.savez(
    TASK_DIR + r"\doe_metrics.npz",
    rmse_c=rmse_cc, eta_c=eta_c, rmse_q=rmse_q, eta_q=eta_q,
    gain=g_c, dx1_um=dx1 * 1e6, dx2_um=dx2 * 1e6, n_sig=n_sig, N=N,
    lam_um=LAM * 1e6, z_mm=Z * 1e3, w0_mm=W0 * 1e3,
    img_mm=IMG_SIZE * 1e3, L2_mm=L2 * 1e3,
)
print("完成")
