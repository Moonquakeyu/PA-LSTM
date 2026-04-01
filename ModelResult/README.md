# Public Course Project (COMP5532) - Streamlit Demo

This repository is a **public code release** for the course project:

**DIGITAL TWINS & VIRTUAL HUMAN (COMP5532)**  
Project title:

**PA-LSTM: Physics-Informed Acceleration-Aware LSTM**  
**for Surgical Trajectory Latency Compensation**

This `ModelResult` folder contains a runnable Streamlit demo for trajectory prediction under communication latency, including RMSE comparison, 3D trajectory visualization, training curves, and ablation study.

## 1. Requirements

- OS: macOS / Linux / Windows
- Python: `3.9+` (recommended `3.10`)
- Packages:
  - `streamlit`
  - `torch`
  - `numpy`
  - `plotly`
  - `scikit-learn`
  - `pandas`

Install dependencies:

```bash
pip install streamlit torch numpy plotly scikit-learn pandas
```

## 2. Required Files

Make sure the following files are in the same directory as `app.py` (this `ModelResult` folder):

- `app.py`
- `artifacts.pkl`
- `vanilla_lstm.pth`
- `attn_lstm.pth`
- `pa_no_pi.pth`
- `pa_proposed.pth`

## 3. Run

From this directory, run:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## 4. Demo Usage

- `Communication Delay`: switch delay setting (`100ms / 200ms / 500ms`)
- `Visualisation window (frames)`: adjust trajectory window length
- Tabs:
  - `RMSE Comparison`
  - `3D Trajectory`
  - `Training Curves`
  - `Ablation Study`

## 5. FAQ

- **File not found (`artifacts.pkl` or `.pth`)**  
  Ensure all required files are in the same folder as `app.py`.

- **Slow loading**  
  First load of `3D Trajectory` runs model inference and may take several seconds.

- **Port occupied**  
  Run with another port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

# Streamlit 演示运行说明

本目录提供了一个可直接运行的 Streamlit 演示页面，用于展示不同通信延迟下的手术轨迹预测效果（含 RMSE 对比、3D 轨迹、训练曲线和消融实验）。

## 1. 运行环境

- 操作系统：macOS / Linux / Windows 均可
- Python：建议 `3.9+`（推荐 `3.10`）
- 依赖包：
  - `streamlit`
  - `torch`
  - `numpy`
  - `plotly`
  - `scikit-learn`
  - `pandas`

可使用以下命令安装依赖：

```bash
pip install streamlit torch numpy plotly scikit-learn pandas
```

## 2. 文件检查

运行前请确认以下文件与 `app.py` 在同一目录（即当前 `ModelResult` 目录）：

- `app.py`
- `artifacts.pkl`
- `vanilla_lstm.pth`
- `attn_lstm.pth`
- `pa_no_pi.pth`
- `pa_proposed.pth`

## 3. 启动方式

在终端进入本目录后运行：

```bash
streamlit run app.py
```

启动后，终端会显示一个本地地址（通常是 `http://localhost:8501`），在浏览器打开即可。

## 4. 页面使用说明

- 左侧 `Communication Delay`：切换延迟档位（`100ms / 200ms / 500ms`）。
- 左侧 `Visualisation window (frames)`：调节可视化窗口长度。
- 顶部 Tab：
  - `RMSE Comparison`：各方法误差对比表与柱状图
  - `3D Trajectory`：3D 轨迹与 XYZ 分轴时间序列
  - `Training Curves`：训练/验证损失曲线
  - `Ablation Study`：带/不带 PI-Loss 的消融分析

## 5. 常见问题

- **报错“找不到 artifacts.pkl 或 .pth 文件”**  
  请确认上述文件均在 `app.py` 同目录。

- **页面加载较慢**  
  首次进入 `3D Trajectory` 时会进行模型推理，等待数秒属正常。

- **端口被占用**  
  可改端口运行：
  ```bash
  streamlit run app.py --server.port 8502
  ```

