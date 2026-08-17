# DOE 相位设计与光场传播分析套件

基于 VirtualLab Fusion 相机探测器导出数据（λ = 1.064 µm）的光场分析，
以及纯相位 DOE（衍射光学元件）设计：高斯光束入射，自由空间传播 50 mm 后
再现 40 mm × 40 mm 目标图像（罗小黑）。

## 系统参数

| 项目 | 参数 |
| --- | --- |
| 波长 λ | 1.064 µm |
| 探测器数据 | 500×500 网格，50 nm 采样，25×25 µm 视场，相位 (rad) |
| 传播分析 | 高斯光束 (w₀ = 5 µm) 叠加相位，Fresnel 传播 20 mm |
| DOE | 1.5×1.5 mm，1536×1536 像素（0.98 µm/px），纯相位 |
| DOE 入射光 | 基模高斯光束，1/e² 半径 w₀ = 0.75 mm |
| 工作距离 | 50 mm（自由空间 Fresnel 衍射） |
| 目标图像 | 40 mm × 40 mm（罗小黑，灰度化） |

## 设计流程（Pipeline）

```
① 数据解析        VirtualLab CSV → 500×500 相位矩阵 (plot_beam_data.py)
      ↓           二维热图 / 三维表面图
② 相位包裹        mod 2π, gray 色图, LaTeX colorbar (propagation_analysis.py)
      ↓
③ 传播分析        高斯光束 × exp(iφ) → Fresnel 单FFT传播 20 mm
      ↓           8192² 零填充细化采样; Strehl/1e²半径/环绕能量/散射晕
④ DOE 设计        信号窗口 IFTA + MRAF (doe_design.py)
      ↓           3 随机重启 × 120 迭代, complex64 加速
⑤ 报告与分发      自包含 HTML 报告 (base64 内嵌图) + SMTP 邮件自动发送
```

## 核心算法

**Fresnel 单 FFT 传播**（输入零填充至 8192×8192，输出采样 52 µm）：

```
U(x′,y′) ∝ FFT{ E₀(x,y)·exp[iπ(x²+y²)/(λz)] },  I = |U|²
```

**DOE 设计 —— 信号窗口约束 IFTA（Gerchberg–Saxton 改进型 + MRAF）**：

```
像面约束:  U′_SW = [m·s·A_t + (1−m)|U|]·e^{i∠U}   (m = 0.8)
           s = √(E_SW/ΣA_t²)   ← 窗内能量归一（关键：目标振幅与光场同尺度）
           窗外自由（吸收残差，提升窗内保真度）
DOE 面约束: E ← A_Gauss·e^{i∠E′}
```

> 经验：目标振幅（0–1 图像尺度）必须按信号窗口能量归一到光场尺度，
> 否则约束会把能量"压出"窗口（反设计）。

## 主要结果

**传播分析（20 mm）**：Strehl 比 0.99998，1/e² 半径 1.360 mm
（理论 1.355 mm），相位环散射产生约 −42 dB 环状晕（r > 3 mm）。

**DOE 设计（50 mm 再现图像）**：

| 指标 | 连续相位 | 256 级量化 (8 bit) |
| --- | --- | --- |
| 重建 RMSE | 0.0014 | 0.0040 |
| 衍射效率 η | 42.7% | 42.7% |

重建像相对目标旋转 180°（傅里叶传播固有性质；预旋转目标可得正立像）。

## 文件结构

```
code/            全部 Python 脚本（ numpy + matplotlib + Pillow ）
  plot_beam_data.py         ① 数据解析 + 2D/3D 相位图
  propagation_analysis.py   ②③ 包裹相位 + 20 mm 传播分析
  doe_design.py             ④ DOE 相位设计 (IFTA+MRAF)
  gen_html_report*.py       ⑤ 自包含 HTML 报告生成器
figures/         各阶段 300 dpi 图件 (phase/ propagation/ doe/)
reports/         自包含 HTML 分析报告（双击离线查看）
data/            doe_phase_data.npz — DOE 相位数据
                 （连续 + 256 级量化两版, 1536×1536, 0.98 µm/px）
```

## 运行环境

Python 3.x + numpy + matplotlib + Pillow。各脚本顶部有绝对路径
（数据文件、输出目录），按需修改后 `python code/xxx.py` 即可。

## 假设与注意事项

- 波长取本批次数据波长 1.064 µm；DOE 相位深度 ∝ 1/λ，换波长需重新设计。
- 高斯光束腰半径为假设值（分析 5 µm / DOE 0.75 mm），脚本中可改。
- 图像边缘衍射角 ≈ 22°，Fresnel 近轴近似误差 ~2%，严格设计可用角谱法。
- DOE 最细局部光栅周期 ≈ 3 像素 ≈ 2.9 µm，电子束/激光直写可制造。
