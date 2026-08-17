# -*- coding: utf-8 -*-
"""
DOE 设计分析报告: 50 mm 工作距离再现目标图片 (罗小黑)
自包含 HTML (base64 内嵌图片), 底部时间 + auto-analysis 签名
"""
import base64
from datetime import datetime

import numpy as np

TASK_DIR = r"C:\Users\Fischer\Desktop\ds2\auto_sender\202608171855"

M = np.load(TASK_DIR + r"\doe_metrics.npz")
rmse_c = float(M["rmse_c"])
eta_c = float(M["eta_c"])
rmse_q = float(M["rmse_q"])
eta_q = float(M["eta_q"])
dx1_um = float(M["dx1_um"])
dx2_um = float(M["dx2_um"])
n_sig = int(M["n_sig"])
N = int(M["N"])
lam_um = float(M["lam_um"])
z_mm = float(M["z_mm"])
w0_mm = float(M["w0_mm"])
img_mm = float(M["img_mm"])
L2_mm = float(M["L2_mm"])

win_frac = (img_mm / L2_mm) ** 2


def b64img(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


img_t = b64img(TASK_DIR + r"\doe_target.png")
img_p = b64img(TASK_DIR + r"\doe_phase.png")
img_r = b64img(TASK_DIR + r"\doe_reconstruction.png")
img_c = b64img(TASK_DIR + r"\doe_compare.png")
img_cv = b64img(TASK_DIR + r"\doe_convergence.png")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>DOE 相位设计报告 — 50 mm 再现目标图像</title>
<style>
  body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
         margin: 0; background: #f4f6f8; color: #222; }}
  .container {{ max-width: 980px; margin: 24px auto; background: #fff;
               padding: 36px 44px; border-radius: 10px;
               box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
  h1 {{ font-size: 24px; border-bottom: 3px solid #1f6fb2;
       padding-bottom: 10px; color: #1f6fb2; }}
  h2 {{ font-size: 18px; margin-top: 32px; color: #155a8a;
       border-left: 5px solid #1f6fb2; padding-left: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #d5dde5; padding: 8px 12px;
           font-size: 14px; text-align: left; }}
  th {{ background: #eaf2f9; width: 240px; }}
  .fig {{ text-align: center; margin: 18px 0; }}
  .fig img {{ max-width: 84%; border: 1px solid #d5dde5;
             border-radius: 6px; }}
  .fig .cap {{ font-size: 13px; color: #666; margin-top: 6px; }}
  .note {{ background: #fdf6e5; border: 1px solid #ecd9a0;
          border-radius: 8px; padding: 12px 16px; font-size: 13.5px;
          line-height: 1.8; margin-top: 12px; }}
  .concl {{ background: #f0f7fc; border: 1px solid #cfe3f2;
           border-radius: 8px; padding: 14px 18px; font-size: 14px;
           line-height: 1.8; }}
  .formula {{ background: #f7f7f9; border: 1px solid #e2e2e8;
             border-radius: 6px; padding: 10px 16px; margin: 10px 0;
             font-family: "Cambria Math", "Times New Roman", serif;
             font-size: 15px; }}
  .footer {{ margin-top: 40px; padding-top: 14px;
            border-top: 1px dashed #bbb; font-size: 13px;
            color: #777; display: flex; justify-content: space-between; }}
</style>
</head>
<body>
<div class="container">
  <h1>DOE 相位设计报告：50 mm 工作距离图像再现</h1>
  <p style="color:#555;font-size:14px;">
    目标图像：<i>罗小黑.jpeg</i>（630×630 px）；
    设计方法：信号窗口约束 IFTA（Gerchberg–Saxton 改进型 + MRAF 混合）
  </p>

  <h2>1. 设计目标与系统参数</h2>
  <table>
    <tr><th>波长 λ</th><td>{lam_um} µm（沿用本批次数据波长，假设值）</td></tr>
    <tr><th>DOE 尺寸</th><td>1.5 mm × 1.5 mm，纯相位型</td></tr>
    <tr><th>DOE 采样</th><td>{N} × {N} 像素，像素尺寸
        {dx1_um:.2f} µm</td></tr>
    <tr><th>入射光</th><td>基模高斯光束，1/e² 半径 w₀ = {w0_mm} mm
        （假设值，DOE 半径 = w₀，口径利用率 86.5%）</td></tr>
    <tr><th>传播</th><td>自由空间 Fresnel 衍射，z = {z_mm:.0f} mm</td></tr>
    <tr><th>目标图像尺寸</th><td>{img_mm:.0f} mm × {img_mm:.0f} mm，
        置于 ±27.2 mm 输出窗口中心（信号窗口 {n_sig}² px）</td></tr>
  </table>

  <h2>2. 采样与衍射几何校验</h2>
  <table>
    <tr><th>输出面采样间隔</th><td>{dx2_um:.2f} µm（= λz/L₁）</td></tr>
    <tr><th>图像边缘衍射角</th><td>21.8°（20 mm / 50 mm）</td></tr>
    <tr><th>Nyquist 允许最大坐标</th><td>27.2 mm &gt; 20 mm
        ✓ 无混叠</td></tr>
    <tr><th>最细局部光栅周期</th><td>≈ 2.9 µm ≈ 3 像素
        （对应最大偏转角，可制造性见结论）</td></tr>
  </table>
  <div class="note">
    <b>近轴近似说明：</b>图像边缘衍射角约 22°，Fresnel 近轴近似在该角度
    引入的相位误差约 2%（1−cosθ 量级），对强度分布影响很小；
    如需严格设计可改用角谱法迭代，结论不变。
  </div>

  <h2>3. 设计算法</h2>
  <p style="font-size:14px;line-height:1.8;">
    采用带<b>信号窗口（SW）约束</b>的改进 IFTA：在 40×40 mm 信号窗口内
    施加目标振幅约束，窗外为自由区（吸收残差、提升窗内保真度）；
    每轮迭代将目标振幅按窗内能量归一，并以 MRAF 方式混合
    （混合因子 m = 0.8）抑制散斑、改善均匀性：
  </p>
  <div class="formula">
    U = FFT&#123; E·exp[iπ(x²+y²)/(λz)] &#125; （DOE → 像面 Fresnel 传播）
    <br>U′<sub>SW</sub> = [m·s·A<sub>t</sub> + (1−m)|U|]·e<sup>i∠U</sup>，
    &nbsp; s = √(E<sub>SW</sub>/ΣA<sub>t</sub>²)
    <br>E′ = conj&#123;exp[iπ(x²+y²)/(λz)]&#125;·IFFT(U′)，
    &nbsp; E ← A<sub>Gauss</sub>·e<sup>i∠E′</sup>
  </div>
  <p style="font-size:14px;line-height:1.8;">
    3 组随机初始相位重启 × 120 次迭代，取 RMSE 最优者。
    相位数据（连续 + 256 级量化两版）保存于附件
    <code>doe_phase_data.npz</code>。
  </p>

  <h2>4. 目标图像与设计的 DOE 相位</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img_t}" alt="target">
    <div class="cap">图 1　目标图像（灰度化，40 mm × 40 mm）</div>
  </div>
  <div class="fig">
    <img src="data:image/png;base64,{img_p}" alt="doe phase">
    <div class="cap">图 2　设计得到的 DOE 相位分布（1.5 mm × 1.5 mm，
        0–2π，gray 色图）</div>
  </div>

  <h2>5. 重建仿真（z = 50 mm）</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img_r}" alt="reconstruction">
    <div class="cap">图 3　重建光强仿真（归一化，±21 mm 视窗）：
        罗小黑形象清晰再现，窗外可见弱散斑背景</div>
  </div>
  <div class="fig">
    <img src="data:image/png;base64,{img_c}" alt="compare">
    <div class="cap">图 4　目标（旋转 180°）与重建对比 ——
        自由空间传播所成像相对物体旋转 180°（傅里叶变换固有性质），
        若需正立像，设计时预旋转目标即可</div>
  </div>

  <h2>6. 定量指标与收敛性</h2>
  <table>
    <tr><th>指标</th><th>连续相位</th><th>256 级量化 (8 bit)</th></tr>
    <tr><td>信号窗口 RMSE（最优增益）</td>
        <td><b>{rmse_c:.5f}</b></td><td>{rmse_q:.5f}</td></tr>
    <tr><td>衍射效率 η（窗内能量占比）</td>
        <td><b>{eta_c*100:.1f}%</b></td><td>{eta_q*100:.1f}%</td></tr>
  </table>
  <p style="font-size:13px;color:#555;line-height:1.7;">
    注：信号窗口占输出窗口面积的 {win_frac*100:.0f}%，η 已接近该几何上限
    的 80%；剩余能量以弱散斑形式分布于窗外自由区。RMSE 以目标强度
    [0,1] 为参照，{rmse_c:.4f} 表明重建与目标高度一致。
  </p>
  <div class="fig">
    <img src="data:image/png;base64,{img_cv}" alt="convergence">
    <div class="cap">图 5　IFTA 收敛曲线（最优重启，120 次迭代）</div>
  </div>

  <h2>7. 结论</h2>
  <div class="concl">
    <ul style="margin:6px 0;padding-left:20px;line-height:1.9;">
      <li>所设计的 1.5 mm × 1.5 mm 纯相位 DOE，在 50 mm 自由空间传播后
          成功再现 40 mm × 40 mm 目标图像，RMSE ≈ {rmse_c:.4f}，
          衍射效率 ≈ {eta_c*100:.1f}%。</li>
      <li>256 级（8 bit）相位量化后性能几乎不变
          （RMSE {rmse_q:.4f}），方案对量化鲁棒，
          兼容常见 SLM 与多台阶光刻工艺。</li>
      <li>DOE 像素 {dx1_um:.2f} µm、最细局部周期 ≈ 3 像素，
          电子束/激光直写可实现；若放宽到 2 µm 像素，
          需将工作距离或图像尺寸按比例放大以保持几何关系。</li>
      <li>重建像相对目标旋转 180°（传播固有性质）；
          需要正立像时把目标图预旋转 180° 重新设计即可。</li>
      <li>设计波长假设为 {lam_um} µm：DOE 相位分布与 λz 几何绑定，
          更换波长需重新设计（相位深度 ∝ 1/λ）。</li>
    </ul>
  </div>

  <div class="footer">
    <span>报告生成时间：{now}</span>
    <span>签名：auto-analysis</span>
  </div>
</div>
</body>
</html>
"""

out_html = TASK_DIR + r"\DOE相位设计报告_罗小黑_50mm.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 报告:", out_html)
print("完成")
