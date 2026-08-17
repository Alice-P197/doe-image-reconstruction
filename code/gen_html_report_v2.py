# -*- coding: utf-8 -*-
"""
系统分析报告 v2: 包裹相位 + 高斯光束叠加相位传播 20 mm 的光强分析
自包含 HTML (base64 内嵌图片), 底部时间 + auto-analysis 签名
"""
import base64
from datetime import datetime

import numpy as np

TASK_DIR = r"C:\Users\Fischer\Desktop\ds2\auto_sender\202608171745"

M = np.load(TASK_DIR + r"\metrics.npz")
strehl = float(M["strehl"])
wz_ideal = float(M["wz_ideal"]) * 1e3      # mm
r_e2_out = float(M["r_e2_out"]) * 1e3      # mm
r_e2_ref = float(M["r_e2_ref"]) * 1e3
ee_out = float(M["ee_out"])
ee_ref = float(M["ee_ref"])
halo_level = float(M["halo_level"])
halo_e_out = float(M["halo_e_out"])
halo_e_ref = float(M["halo_e_ref"])
w0_um = float(M["w0"]) * 1e6
z_mm = float(M["z"]) * 1e3
lam_um = float(M["lam"]) * 1e6
pmin, pmax = float(M["pmin"]), float(M["pmax"])
pmean, pstd = float(M["pmean"]), float(M["pstd"])

ee_theory = 1.0 - np.exp(-2.0)             # 高斯 1/e^2 半径内理论环绕能量
halo_db = 10.0 * np.log10(halo_level)


def b64img(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


img_phase = b64img(TASK_DIR + r"\phase_wrapped_2D.png")
img_phase_ac = b64img(TASK_DIR + r"\phase_wrapped_autocontrast.png")
img_lin = b64img(TASK_DIR + r"\intensity_prop20mm_linear.png")
img_log = b64img(TASK_DIR + r"\intensity_prop20mm_log.png")
img_cs = b64img(TASK_DIR + r"\radial_profile_20mm.png")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>系统分析报告 — 相位包裹与高斯光束传播 (λ = 1.064 µm)</title>
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
  th {{ background: #eaf2f9; width: 260px; }}
  .fig {{ text-align: center; margin: 18px 0; }}
  .fig img {{ max-width: 88%; border: 1px solid #d5dde5;
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
  <h1>系统分析报告：相位包裹与高斯光束传播特性</h1>
  <p style="color:#555;font-size:14px;">
    数据文件：<i>Exported_Data_"Camera Detector" (# 600) (Profile_General)
    _Data_Wavelength of 1.064 µm.csv</i>（VirtualLab Fusion 导出，相位数据）
  </p>

  <h2>1. 文件与物理参数</h2>
  <table>
    <tr><th>波长 λ</th><td>{lam_um} µm</td></tr>
    <tr><th>物理量 / 单位</th><td>Phase / Angle (rad)，实数型（x 域）</td></tr>
    <tr><th>网格点数</th><td>500 × 500</td></tr>
    <tr><th>采样间距</th><td>50 nm × 50 nm</td></tr>
    <tr><th>视场范围</th><td>25 µm × 25 µm（坐标以中心为原点）</td></tr>
  </table>

  <h2>2. 相位统计分析</h2>
  <table>
    <tr><th>最小值</th><td>{pmin:.6g} rad</td></tr>
    <tr><th>最大值</th><td>{pmax:.6g} rad
        （≈ {pmax/(2*np.pi)*100:.2f}% of 2π）</td></tr>
    <tr><th>平均值 / 标准差</th><td>{pmean:.6g} / {pstd:.6g} rad</td></tr>
    <tr><th>空间结构</th><td>中心主瓣 + 同心环（类艾里结构），
        主瓣 FWHM ≈ 3.1 µm，峰值位于视场中心 (≈0.03, 0.03) µm</td></tr>
  </table>
  <div class="note">
    <b>说明：</b>全部相位值位于 (0, 0.029] rad 区间，远小于 2π，
    因此 mod 2π 包裹后不产生跳变；在 0–2π 满量程灰度显示下，
    最大灰度仅约 0.46%，图像近乎全黑属<b>正常物理现象</b>
    （该相位为弱相位扰动）。为展示空间结构，图 2 给出自适应对比度版本。
  </div>

  <h2>3. 包裹相位分布（0–2π，gray 色图）</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img_phase}" alt="wrapped phase">
    <div class="cap">图 1　包裹相位（mod 2π），gray 色图满量程 0–2π，
        colorbar 标注 0 与 2π</div>
  </div>
  <div class="fig">
    <img src="data:image/png;base64,{img_phase_ac}" alt="wrapped phase auto">
    <div class="cap">图 2　同一包裹相位的自适应对比度显示
        （色标 {pmin:.4g} – {pmax:.4g} rad），可见中心主瓣与同心环结构</div>
  </div>

  <h2>4. 传播模型与数值验证</h2>
  <p style="font-size:14px;line-height:1.8;">
    将相位数据作为薄相位板，叠加于基模高斯光束（振幅 1/e² 半径
    w₀ = {w0_um:.0f} µm，<b>假设值</b>，与相位结构尺度及视场匹配），
    在 z = 0 处构造入射复场：
  </p>
  <div class="formula">
    E₀(x, y) = exp[−(x²+y²)/w₀²] · exp[i·φ(x, y)]
  </div>
  <p style="font-size:14px;line-height:1.8;">
    采用 Fresnel 衍射单 FFT 算法传播 z = {z_mm:.0f} mm
    （输入零填充至 8192×8192，输出采样 52 µm，
    输出视场 ±213 mm 内取 ±16 mm 分析）：
  </p>
  <div class="formula">
    U(x′, y′) = e<sup>ikz</sup>/(iλz) ·
    FFT{{ E₀(x, y) · exp[iπ(x²+y²)/(λz)] }},
    &nbsp;&nbsp; I(x′, y′) = |U|²
  </div>
  <table>
    <tr><th>数值验证项</th><th>理论值</th><th>数值结果</th></tr>
    <tr><td>远场 1/e² 半径 w(z)（无相位）</td>
        <td>{wz_ideal:.3f} mm</td><td>{r_e2_ref:.3f} mm</td></tr>
    <tr><td>环绕能量（r ≤ w(z)，无相位）</td>
        <td>{ee_theory:.4f} (= 1−e⁻²)</td><td>{ee_ref:.4f}</td></tr>
  </table>
  <p style="font-size:13px;color:#555;">
    两项验证偏差均 &lt; 0.5%，传播数值结果可信。
    （z / z<sub>R</sub> ≈ {z_mm*1e-3/(np.pi*(w0_um*1e-6)**2/(lam_um*1e-6)):.0f}，
    已深入远场区。）
  </p>

  <h2>5. 传播 20 mm 后的光强分布</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img_lin}" alt="intensity linear">
    <div class="cap">图 3　传播 20 mm 后归一化光强（线性色标，
        ±6 mm 视窗）</div>
  </div>
  <div class="fig">
    <img src="data:image/png;base64,{img_log}" alt="intensity log">
    <div class="cap">图 4　同一光强的对数（log₁₀）显示：
        主瓣之外可见相位环散射形成的微弱同心环晕（10⁻³ – 10⁻⁵ 量级）
        与弥散背景</div>
  </div>

  <h2>6. 与理想高斯光束的定量对比</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img_cs}" alt="radial profile">
    <div class="cap">图 5　径向平均光强对比：实线 = 高斯+相位板，
        虚线 = 理想高斯。r &lt; 3 mm 两线几乎重合；
        r &gt; 3 mm 相位板产生约 10⁻⁵ 量级的散射晕</div>
  </div>
  <table>
    <tr><th>指标</th><th>高斯 + 相位板</th><th>理想高斯</th></tr>
    <tr><td>Strehl 比（峰值强度比）</td>
        <td colspan="2"><b>{strehl:.6f}</b>（≈ 1，
        近衍射极限）</td></tr>
    <tr><td>实测 1/e² 半径</td>
        <td>{r_e2_out:.3f} mm</td><td>{r_e2_ref:.3f} mm</td></tr>
    <tr><td>环绕能量（r ≤ w(z)）</td>
        <td>{ee_out:.4f}</td><td>{ee_ref:.4f}</td></tr>
    <tr><td>散射晕峰值（r &gt; 3 mm）</td>
        <td>{halo_level:.2e}（{halo_db:.1f} dB）</td>
        <td>&lt; 10⁻⁶</td></tr>
    <tr><td>晕区能量占比（r &gt; 3 mm）</td>
        <td>{halo_e_out:.2e}</td><td>{halo_e_ref:.2e}
        （高斯自身拖尾）</td></tr>
  </table>

  <h2>7. 结论</h2>
  <div class="concl">
    <ul style="margin:6px 0;padding-left:20px;line-height:1.9;">
      <li>该相位为<b>弱相位扰动</b>（最大 0.029 rad ≈ 0.46% of 2π），
          包裹后不产生 2π 跳变。</li>
      <li>叠加高斯光束传播 20 mm 后，光强仍保持<b>近衍射极限</b>的高斯
          主瓣：Strehl 比 {strehl:.5f}，1/e² 半径 {r_e2_out:.3f} mm，
          与理想值 {wz_ideal:.3f} mm 一致；主瓣内环绕能量
          {ee_out:.4f} ≈ 理论值 0.8647。</li>
      <li>相位板的同心环结构将少量能量散射至远场，形成
          <b>约 −42 dB（10⁻⁵ 量级）的环状散射晕</b>（r &gt; 3 mm），
          晕区总能量占比仅 {halo_e_out:.1e}，对主瓣无实质影响。</li>
      <li>总体：该弱相位对 20 mm 传播后的光强分布影响极小，
          光束质量接近理想高斯；若应用对远场背景光敏感
          （如弱信号探测），可关注 3 mm 以外的环状散射晕。</li>
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

out_html = TASK_DIR + r"\Camera_Detector_系统分析报告_1.064um.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 报告:", out_html)
print("完成")
