# -*- coding: utf-8 -*-
"""
高斯光束叠加相位板 -> Fresnel 传播 20 mm -> 光强分布系统分析
输出:
  1) 包裹相位图 (0-2pi, gray, LaTeX colorbar)
  2) 传播后光强 (线性, 对数)
  3) 径向截面对比 (有/无相位)
  4) 指标: Strehl 比, 1/e^2 半径, 环绕能量, 最大旁瓣
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"

TASK_DIR = r"C:\Users\Fischer\Desktop\ds2\auto_sender\202608171745"

# ---------------- 读取相位数据 ----------------
path = glob.glob(
    r"C:\Users\Fischer\Desktop\数据\强度\csv格式\csv\Exported_Data_*1.064*.csv"
)[0]
phase = np.loadtxt(path, delimiter=";", skiprows=7, encoding="utf-8")
N = phase.shape[0]
dx = 50e-9                       # 50 nm
lam = 1.064e-6                   # 1.064 µm
k = 2.0 * np.pi / lam
z = 20e-3                        # 20 mm

axis0 = (np.arange(N) - N // 2) * dx          # m
X0, Y0 = np.meshgrid(axis0, axis0)
R0 = np.sqrt(X0**2 + Y0**2)

# ---------------- 1) 包裹相位图 ----------------
phase_wrapped = np.mod(phase, 2.0 * np.pi)

fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)
im = ax.imshow(
    phase_wrapped,
    extent=[axis0[0] * 1e6, axis0[-1] * 1e6] * 2,
    origin="lower", cmap="gray", vmin=0.0, vmax=2.0 * np.pi,
    aspect="equal",
)
ax.set_xlabel("x (µm)", fontsize=13)
ax.set_ylabel("y (µm)", fontsize=13)
ax.set_title("Wrapped Phase @ λ = 1.064 µm", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0.0, 2.0 * np.pi])
cb.set_ticklabels([r"$0$", r"$2\pi$"], fontsize=13)
cb.set_label("Phase (rad)", fontsize=12)
fig.tight_layout()
f_phase = TASK_DIR + r"\phase_wrapped_2D.png"
fig.savefig(f_phase, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f_phase)

# 自适应对比度版本 (结构可见, 供报告补充)
fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)
im = ax.imshow(
    phase_wrapped,
    extent=[axis0[0] * 1e6, axis0[-1] * 1e6] * 2,
    origin="lower", cmap="gray",
    vmin=float(phase_wrapped.min()), vmax=float(phase_wrapped.max()),
    aspect="equal",
)
ax.set_xlabel("x (µm)", fontsize=13)
ax.set_ylabel("y (µm)", fontsize=13)
ax.set_title("Wrapped Phase (auto contrast)", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Phase (rad)", fontsize=12)
fig.tight_layout()
f_phase_ac = TASK_DIR + r"\phase_wrapped_autocontrast.png"
fig.savefig(f_phase_ac, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f_phase_ac)

# ---------------- 2) 入射场: 高斯 x 相位板 ----------------
w0 = 5e-6                        # 高斯光束腰半径 5 µm (假设, 报告中注明)
gauss = np.exp(-(R0 / w0) ** 2)  # 振幅, 1/e^2 强度半径 = w0
E_in = gauss * np.exp(1j * phase)
E_ref = gauss                    # 参考: 无相位

# ---------------- 3) Fresnel 单 FFT 传播 (零填充细化输出采样) ----------------
NPAD = 8192                      # 输出采样 dx2 = lam*z/(NPAD*dx) ~ 52 µm

def fresnel_prop(E, z, lam, dx, npad):
    Nn = E.shape[0]
    ax_ = (np.arange(Nn) - Nn // 2) * dx
    X_, Y_ = np.meshgrid(ax_, ax_)
    q = np.exp(1j * np.pi / (lam * z) * (X_**2 + Y_**2))
    Eq = E * q
    pad = (npad - Nn) // 2
    Eq = np.pad(Eq, pad)
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Eq)))
    dx2 = lam * z / (npad * dx)
    ax2 = (np.arange(npad) - npad // 2) * dx2
    return U, ax2

print("传播计算中 (8192x8192 FFT)...")
U_out, axis1 = fresnel_prop(E_in, z, lam, dx, NPAD)
U_ref, _ = fresnel_prop(E_ref, z, lam, dx, NPAD)
I_out = np.abs(U_out) ** 2
I_ref = np.abs(U_ref) ** 2
del U_out, U_ref

# 裁剪到 ±16 mm 再算指标 (远处能量可忽略)
m16 = np.abs(axis1) <= 16e-3
axis_c = axis1[m16]
I_out = I_out[np.ix_(m16, m16)]
I_ref = I_ref[np.ix_(m16, m16)]

strehl = I_out.max() / I_ref.max()
In = I_out / I_out.max()
Irn = I_ref / I_ref.max()
X1, Y1 = np.meshgrid(axis_c, axis_c)
R1 = np.sqrt(X1**2 + Y1**2)

# ---------------- 4) 指标 ----------------
# 理想高斯远场 1/e^2 半径
zR = np.pi * w0**2 / lam
wz_ideal = w0 * np.sqrt(1.0 + (z / zR) ** 2)

# 径向平均 -> 实测 1/e^2 半径
def radial_profile(I2d, R, rmax, nb=2000):
    m = R <= rmax
    r_flat = R[m].ravel()
    i_flat = I2d[m].ravel()
    order = np.argsort(r_flat)
    r_s, i_s = r_flat[order], i_flat[order]
    edges = np.linspace(0, rmax, nb + 1)
    idx = np.searchsorted(r_s, edges)
    r_c, i_m = [], []
    for b in range(nb):
        if idx[b + 1] > idx[b]:
            s = slice(idx[b], idx[b + 1])
            r_c.append(r_s[s].mean())
            i_m.append(i_s[s].mean())
    return np.array(r_c), np.array(i_m)

r_c, prof_out = radial_profile(In, R1, 8e-3)
_, prof_ref = radial_profile(Irn, R1, 8e-3)

def e2_radius(r, p):
    thr = 1.0 / np.e**2
    j = np.where(p < thr)[0]
    return float(r[j[0]]) if len(j) else float("nan")

r_e2_out = e2_radius(r_c, prof_out)
r_e2_ref = e2_radius(r_c, prof_ref)

# 环绕能量 (在 r = wz_ideal 内)
def encircled(I2d, R, rmax):
    m = R <= rmax
    return float(I2d[m].sum() / I2d.sum())

ee_out = encircled(In, R1, wz_ideal)
ee_ref = encircled(Irn, R1, wz_ideal)

# 散射晕指标 (r > 3 mm, 此处理想高斯已 < 1e-4)
HALO_R = 3e-3
m_halo = r_c > HALO_R * 1e3 / 1e3  # r_c 单位为 m
m_halo = r_c > HALO_R
halo_level = float(prof_out[m_halo].max()) if m_halo.any() else float("nan")
halo_e_out = float(In[R1 > HALO_R].sum() / In.sum())
halo_e_ref = float(Irn[R1 > HALO_R].sum() / Irn.sum())

print("Strehl = %.6f" % strehl)
print("理想 w(z) = %.3f mm" % (wz_ideal * 1e3))
print("实测 1/e^2 半径: 有相位 %.3f mm / 无相位 %.3f mm"
      % (r_e2_out * 1e3, r_e2_ref * 1e3))
print("环绕能量@w(z): 有相位 %.4f / 无相位 %.4f" % (ee_out, ee_ref))
print("散射晕峰值 (r>3mm) = %.3e (%.1f dB)"
      % (halo_level, 10 * np.log10(halo_level)))
print("散射晕能量占比: 有相位 %.3e / 无相位 %.3e" % (halo_e_out, halo_e_ref))

# ---------------- 5) 光强图 (裁剪 ±6 mm) ----------------
crop = 6e-3
mc = (np.abs(axis_c) <= crop)
I2d_lin = In[np.ix_(mc, mc)]
I2d_log = np.log10(np.maximum(In, 1e-9))[np.ix_(mc, mc)]
ext = [axis_c[mc][0] * 1e3, axis_c[mc][-1] * 1e3] * 2

fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)
im = ax.imshow(I2d_lin, extent=ext, origin="lower", cmap="inferno",
               aspect="equal", vmin=0, vmax=1)
ax.set_xlabel("x (mm)", fontsize=13)
ax.set_ylabel("y (mm)", fontsize=13)
ax.set_title("Intensity after 20 mm propagation (linear)", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Normalized intensity", fontsize=12)
fig.tight_layout()
f_lin = TASK_DIR + r"\intensity_prop20mm_linear.png"
fig.savefig(f_lin, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f_lin)

fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)
im = ax.imshow(I2d_log, extent=ext, origin="lower", cmap="inferno",
               aspect="equal", vmin=-6, vmax=0)
ax.set_xlabel("x (mm)", fontsize=13)
ax.set_ylabel("y (mm)", fontsize=13)
ax.set_title("Intensity after 20 mm propagation (log$_{10}$)", fontsize=14)
cb = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0, -2, -4, -6])
cb.set_label(r"$\log_{10}(I/I_{max})$", fontsize=12)
fig.tight_layout()
f_log = TASK_DIR + r"\intensity_prop20mm_log.png"
fig.savefig(f_log, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f_log)

# ---------------- 6) 径向截面对比 ----------------
fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=300)
ax.semilogy(r_c * 1e3, prof_out, lw=1.6, color="#c0362c",
            label="Gaussian + phase mask")
ax.semilogy(r_c * 1e3, prof_ref, lw=1.6, color="#1f6fb2",
            ls="--", label="Ideal Gaussian (no phase)")
ax.axhline(1 / np.e**2, color="gray", ls=":", lw=1)
ax.text(4.7, 1 / np.e**2 * 1.25, r"$1/e^2$", fontsize=11, color="gray")
ax.set_xlim(0, 6)
ax.set_ylim(1e-6, 2)
ax.set_xlabel("r (mm)", fontsize=13)
ax.set_ylabel("Normalized intensity (log)", fontsize=13)
ax.set_title("Radial intensity profile @ z = 20 mm", fontsize=14)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11)
fig.tight_layout()
f_cs = TASK_DIR + r"\radial_profile_20mm.png"
fig.savefig(f_cs, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已保存:", f_cs)

# 指标存盘供报告脚本使用
np.savez(
    TASK_DIR + r"\metrics.npz",
    strehl=strehl, wz_ideal=wz_ideal, r_e2_out=r_e2_out,
    r_e2_ref=r_e2_ref, ee_out=ee_out, ee_ref=ee_ref,
    halo_level=halo_level, halo_e_out=halo_e_out,
    halo_e_ref=halo_e_ref, w0=w0, z=z, lam=lam,
    pmin=float(phase.min()), pmax=float(phase.max()),
    pmean=float(phase.mean()), pstd=float(phase.std()),
)
print("完成")
