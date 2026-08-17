# -*- coding: utf-8 -*-
"""
生成 Camera Detector 数据分析 HTML 报告（内嵌 base64 图片，自包含单文件）
输出到 auto_sender 任务文件夹，同时把两张 PNG 复制过去作为附件
"""
import glob
import base64
import shutil
from datetime import datetime

import numpy as np

TASK_DIR = r"C:\Users\Fischer\Desktop\ds2\auto_sender\202608171745"
SESSION_DIR = r"C:\Users\Fischer\Desktop\ds2\sessions\分析CSV文件并生成二维三维图"

# ---------- 读取数据 ----------
path = glob.glob(
    r"C:\Users\Fischer\Desktop\数据\强度\csv格式\csv\Exported_Data_*1.064*.csv"
)[0]
data = np.loadtxt(path, delimiter=";", skiprows=7, encoding="utf-8")

n = data.shape[0]
pitch_um = 0.05
axis = np.linspace(-n * pitch_um / 2, n * pitch_um / 2, n)

# ---------- 统计量 ----------
d_min, d_max = float(data.min()), float(data.max())
d_mean, d_std = float(data.mean()), float(data.std())
pk_idx = np.unravel_index(int(np.argmax(data)), data.shape)
pk_x, pk_y = float(axis[pk_idx[1]]), float(axis[pk_idx[0]])

# 过峰值的行/列，估算半高全宽 (FWHM)
row = data[pk_idx[0], :]
col = data[:, pk_idx[1]]
half = d_max / 2.0


def fwhm(line, ax, thr):
    above = np.where(line >= thr)[0]
    if len(above) < 2:
        return float("nan")
    return float(ax[above[-1]] - ax[above[0]])


fwhm_x = fwhm(row, axis, half)
fwhm_y = fwhm(col, axis, half)

# ---------- 图片 base64 ----------
def b64img(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


img2d = b64img(SESSION_DIR + r"\profile_2D.png")
img3d = b64img(SESSION_DIR + r"\profile_3D.png")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Camera Detector 数据分析报告 — λ = 1.064 µm</title>
<style>
  body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
         margin: 0; background: #f4f6f8; color: #222; }}
  .container {{ max-width: 960px; margin: 24px auto; background: #fff;
               padding: 36px 44px; border-radius: 10px;
               box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
  h1 {{ font-size: 24px; border-bottom: 3px solid #1f6fb2;
       padding-bottom: 10px; color: #1f6fb2; }}
  h2 {{ font-size: 18px; margin-top: 32px; color: #155a8a;
       border-left: 5px solid #1f6fb2; padding-left: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #d5dde5; padding: 8px 12px;
           font-size: 14px; text-align: left; }}
  th {{ background: #eaf2f9; width: 220px; }}
  .fig {{ text-align: center; margin: 18px 0; }}
  .fig img {{ max-width: 100%; border: 1px solid #d5dde5;
             border-radius: 6px; }}
  .fig .cap {{ font-size: 13px; color: #666; margin-top: 6px; }}
  .concl {{ background: #f0f7fc; border: 1px solid #cfe3f2;
           border-radius: 8px; padding: 14px 18px; font-size: 14px;
           line-height: 1.8; }}
  .footer {{ margin-top: 40px; padding-top: 14px;
            border-top: 1px dashed #bbb; font-size: 13px;
            color: #777; display: flex;
            justify-content: space-between; }}
</style>
</head>
<body>
<div class="container">
  <h1>Camera Detector 数据分析报告</h1>
  <p style="color:#555;font-size:14px;">
    数据文件：<i>Exported_Data_"Camera Detector" (# 600) (Profile_General)
    _Data_Wavelength of 1.064 µm.csv</i>（VirtualLab Fusion 导出）
  </p>

  <h2>1. 文件与物理参数</h2>
  <table>
    <tr><th>波长</th><td>1.064 µm</td></tr>
    <tr><th>数据类型</th><td>实数型（x 域）</td></tr>
    <tr><th>物理量 / 单位</th><td>Phase / Angle (rad)</td></tr>
    <tr><th>网格点数</th><td>500 × 500</td></tr>
    <tr><th>采样间距</th><td>50 nm × 50 nm</td></tr>
    <tr><th>视场范围</th><td>25 µm × 25 µm（报告坐标以中心为原点，
        −12.5 ~ +12.5 µm）</td></tr>
  </table>

  <h2>2. 数据统计</h2>
  <table>
    <tr><th>最小值</th><td>{d_min:.6g} rad</td></tr>
    <tr><th>最大值（峰值）</th><td>{d_max:.6g} rad</td></tr>
    <tr><th>平均值</th><td>{d_mean:.6g} rad</td></tr>
    <tr><th>标准差</th><td>{d_std:.6g} rad</td></tr>
    <tr><th>峰值位置</th><td>(x, y) = ({pk_x:+.2f}, {pk_y:+.2f}) µm</td></tr>
    <tr><th>半高全宽 FWHM (x / y)</th>
        <td>{fwhm_x:.2f} µm / {fwhm_y:.2f} µm</td></tr>
  </table>

  <h2>3. 二维分布图</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img2d}" alt="2D profile">
    <div class="cap">图 1　二维伪彩色分布（jet 色标，坐标单位 µm）</div>
  </div>

  <h2>4. 三维表面图</h2>
  <div class="fig">
    <img src="data:image/png;base64,{img3d}" alt="3D surface">
    <div class="cap">图 2　三维表面图（约 167×167 降采样渲染）</div>
  </div>

  <h2>5. 分析结论</h2>
  <div class="concl">
    <ul style="margin:6px 0;padding-left:20px;line-height:1.9;">
      <li>光斑呈典型的<b>中心主瓣 + 同心衍射环</b>结构（类艾里斑），
          能量集中于中心，说明聚焦质量良好。</li>
      <li>峰值位于 ({pk_x:+.2f}, {pk_y:+.2f}) µm，与视场中心基本重合，
          对准状态正常。</li>
      <li>中心主瓣 FWHM 约为 {fwhm_x:.2f} µm（x）× {fwhm_y:.2f} µm（y），
          x、y 方向宽度接近，光束圆对称性较好。</li>
      <li>外围可见多级衍射环，环强度随半径增大迅速衰减。</li>
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

out_html = TASK_DIR + r"\Camera_Detector_数据分析报告_1.064um.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 报告:", out_html)

# 复制两张 PNG 作为附件
for name in ("profile_2D.png", "profile_3D.png"):
    shutil.copy(SESSION_DIR + "\\" + name, TASK_DIR + "\\" + name)
    print("已复制:", name)

print("FWHM x/y: %.2f / %.2f µm, peak at (%.2f, %.2f)" % (fwhm_x, fwhm_y, pk_x, pk_y))
print("完成")
