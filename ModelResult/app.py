"""
Surgical Robot Teleoperation — Latency Compensation Demo
运行方式：streamlit run app.py
依赖：streamlit, torch, numpy, plotly, scikit-learn, pandas
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surgical Robot Latency Compensation",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 模型定义（必须与训练时完全一致）
# ─────────────────────────────────────────────────────────────────────────────

class VanillaLSTM(nn.Module):
    def __init__(self, h_dim, n_layer, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(3, h_dim, n_layer, batch_first=True, dropout=0.4 if n_layer > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),   # 128→64
            nn.ReLU(),
            nn.Linear(h_dim // 2, pred_len * 3),  # 64→45
        )

    def forward(self, x_pos, x_dyn, acc_w):
        out, hidden = self.lstm(x_pos)
        pred = self.head(out[:, -1, :]).reshape(-1, self.pred_len, 3)
        return pred, hidden


class StandardAttnLSTM(nn.Module):
    def __init__(self, h_dim, n_layer, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.lstm  = nn.LSTM(3, h_dim, n_layer, batch_first=True, dropout=0.4 if n_layer > 1 else 0.0)
        self.attn_v = nn.Linear(h_dim, 1)
        self.head   = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),        # 128→64
            nn.GELU(),                            # 训练时用的是 GELU
            nn.Linear(h_dim // 2, pred_len * 3), # 64→45
        )

    def forward(self, x_pos, x_dyn, acc_w):
        out, hidden = self.lstm(x_pos)                          # (B, T, H)
        e = self.attn_v(torch.tanh(out)).squeeze(-1)            # (B, T)
        alpha = torch.softmax(e, dim=1)                         # (B, T)
        ctx = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)     # (B, H)
        pred = self.head(ctx).reshape(-1, self.pred_len, 3)
        return pred, hidden


class AATA(nn.Module):
    """Acceleration-Aware Temporal Attention — 与训练代码完全一致。"""
    def __init__(self, h_dim):
        super().__init__()
        self.W_h  = nn.Linear(h_dim, h_dim, bias=False)
        self.W_d  = nn.Linear(h_dim, h_dim, bias=False)
        self.W_a  = nn.Linear(h_dim, 1,     bias=True)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, H_pos, H_dyn, acc_w):
        import torch.nn.functional as F
        energy   = self.W_a(torch.tanh(self.W_h(H_pos) + self.W_d(H_dyn))).squeeze(-1)  # (B,T)
        acc_norm = acc_w / (acc_w.max(dim=1, keepdim=True).values.clamp(min=1e-6))
        beta_pos = F.softplus(self.beta)
        alpha    = F.softmax(energy * (1.0 + beta_pos * acc_norm), dim=1)                # (B,T)
        context  = torch.bmm(alpha.unsqueeze(1), H_pos).squeeze(1)                       # (B,H)
        return context, alpha


class PALSTMFused(nn.Module):
    """
    Fused single-stream PA-LSTM — 与训练代码 Option A 完全一致。
    input_proj: Linear(14→h_dim) → GELU → Dropout
    lstm: 单个 LSTM(h_dim→h_dim)
    norm: LayerNorm
    aata: AATA 模块
    head: Linear(2h→h) → GELU → Dropout → Linear(h→pred_len*3)
          即 head.0 和 head.3 是 Linear
    """
    def __init__(self, pos_dim=3, dyn_dim=11, h_dim=64,
                 n_layer=2, pred_len=15, dropout=0.4):
        super().__init__()
        import torch.nn.functional as F
        self.pred_len  = pred_len
        self.h_dim     = h_dim
        input_dim      = pos_dim + dyn_dim   # 14

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(h_dim, h_dim, n_layer,
                            batch_first=True,
                            dropout=dropout if n_layer > 1 else 0.0)
        self.norm = nn.LayerNorm(h_dim)
        self.aata = AATA(h_dim)
        self.head = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),     # head.0
            nn.GELU(),                        # head.1
            nn.Dropout(dropout * 0.5),        # head.2
            nn.Linear(h_dim, pred_len * 3),   # head.3
        )

    def forward(self, x_pos, x_dyn, acc_w):
        x_fused = torch.cat([x_pos, x_dyn], dim=-1)   # (B, T, 14)
        x_proj  = self.input_proj(x_fused)             # (B, T, H)
        H, _    = self.lstm(x_proj)                    # (B, T, H)
        H       = self.norm(H)
        context, alpha = self.aata(H, H, acc_w)        # (B, H), (B, T)
        last_h  = H[:, -1, :]
        fused   = torch.cat([context, last_h], dim=-1) # (B, 2H)
        out     = self.head(fused)                     # (B, pred_len*3)
        pred    = out.view(-1, self.pred_len, 3)
        return pred, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter3D:
    def __init__(self, q_pos=1e-4, q_vel=1e-3, r_obs=1e-2):
        self.initialized = False
        self.x  = np.zeros(6)          # [px,py,pz, vx,vy,vz]
        self.P  = np.eye(6) * 1.0
        dt = 1.0
        self.F  = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H  = np.zeros((3, 6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.Q  = np.diag([q_pos]*3 + [q_vel]*3)
        self.R  = np.eye(3) * r_obs

    def update(self, z):
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def predict_ahead(self, n):
        Fn = np.linalg.matrix_power(self.F, n)
        return (Fn @ self.x)[:3]


def run_kalman(traj, delay_steps):
    kf    = KalmanFilter3D()
    preds = np.zeros_like(traj)
    for i, z in enumerate(traj):
        kf.update(z)
        preds[i] = kf.predict_ahead(delay_steps)
    return preds


def _build_dyn_and_acc(seq):
    """
    从位置序列构造 PA-LSTM 所需的动态特征(11维)与加速度权重。
    这里使用轻量近似：速度、加速度及其派生量，不依赖训练 DataLoader。
    """
    vel = np.diff(seq, axis=0, prepend=seq[0:1])          # (T, 3)
    acc = np.diff(vel, axis=0, prepend=vel[0:1])          # (T, 3)

    speed = np.linalg.norm(vel, axis=1, keepdims=True)    # (T, 1)
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True) # (T, 1)
    jerk = np.diff(acc_norm, axis=0, prepend=acc_norm[0:1])  # (T, 1)
    dyn = np.concatenate([vel, acc, speed, acc_norm, jerk, seq], axis=1)  # 3+3+1+1+1+3=12
    dyn = dyn[:, :11].astype(np.float32)                  # 对齐训练输入维度
    acc_w = (acc_norm[:, 0] + 1e-6).astype(np.float32)    # (T,)
    return dyn, acc_w


def run_lstm_trajectory(model, traj, seq_len, pred_len):
    """
    使用滑窗推理得到与原轨迹同长度的 one-step 预测轨迹。
    模型输出 pred_len 帧，这里取第 1 帧作为当前时刻预测值。
    """
    traj = np.asarray(traj, dtype=np.float32)
    total_len = len(traj)
    preds = np.copy(traj)

    for t in range(seq_len, total_len):
        x_pos_np = traj[t - seq_len:t]  # (seq_len, 3)
        x_pos = torch.from_numpy(x_pos_np).unsqueeze(0)   # (1, T, 3)

        if isinstance(model, PALSTMFused):
            x_dyn_np, acc_w_np = _build_dyn_and_acc(x_pos_np)
            x_dyn = torch.from_numpy(x_dyn_np).unsqueeze(0)              # (1, T, 11)
            acc_w = torch.from_numpy(acc_w_np).unsqueeze(0)              # (1, T)
        else:
            x_dyn = torch.zeros((1, seq_len, 11), dtype=torch.float32)
            acc_w = torch.ones((1, seq_len), dtype=torch.float32)

        with torch.no_grad():
            pred, _ = model(x_pos, x_dyn, acc_w)                         # (1, pred_len, 3)
        preds[t] = pred[0, 0].cpu().numpy()

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# 数据 & 模型加载（缓存，只加载一次）
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_all():
    device = torch.device('cpu')
    H_DIM, N_LAYER, PRED_LEN, DROPOUT = 64, 2, 15, 0.4

    # ── 加载 pkl ──
    with open('./artifacts.pkl', 'rb') as f:
        art = pickle.load(f)

    pos_scaler        = art['pos_scaler']
    traj_test_n       = art['traj_test_n']
    results_by_option = art['results_by_option']
    res_delayed       = art['res_delayed']
    shared_histories  = art['shared_histories']
    pa_histories      = art['pa_histories']
    delay_labels      = art.get('delay_labels', ['100ms', '200ms', '500ms'])
    delay_steps       = art.get('delay_steps',  [3, 6, 15])

    # ── 重建模型并加载权重 ──
    models = {
        'Vanilla LSTM':          VanillaLSTM(128, N_LAYER, PRED_LEN),
        'Attention-LSTM':        StandardAttnLSTM(128, N_LAYER, PRED_LEN),
        'PA-LSTM (w/o PI-Loss)': PALSTMFused(3, 11, H_DIM, N_LAYER, PRED_LEN, DROPOUT),
        'PA-LSTM (Proposed)':    PALSTMFused(3, 11, H_DIM, N_LAYER, PRED_LEN, DROPOUT),
    }
    weight_files = {
        'Vanilla LSTM':          'vanilla_lstm.pth',
        'Attention-LSTM':        'attn_lstm.pth',
        'PA-LSTM (w/o PI-Loss)': 'pa_no_pi.pth',
        'PA-LSTM (Proposed)':    'pa_proposed.pth',
    }
    for name, mdl in models.items():
        mdl.load_state_dict(torch.load(weight_files[name], map_location=device))
        mdl.eval()

    return {
        'models':             models,
        'pos_scaler':         pos_scaler,
        'traj_test_n':        traj_test_n,
        'results_by_option':  results_by_option,
        'res_delayed':        res_delayed,
        'shared_histories':   shared_histories,
        'pa_histories':       pa_histories,
        'delay_labels':       delay_labels,
        'delay_steps':        delay_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 Surgical Robot Teleoperation — Latency Compensation")
st.markdown(
    "Physics-Aware LSTM with PI-Loss for surgical trajectory prediction under communication delay."
)

# 加载数据
with st.spinner("Loading models and data..."):
    try:
        data = load_all()
    except FileNotFoundError as e:
        st.error(f"❌ 找不到文件：{e}\n\n请确保 `artifacts.pkl` 和 `.pth` 文件与 `app.py` 在同一目录下。")
        st.stop()

models            = data['models']
pos_scaler        = data['pos_scaler']
traj_test_n       = data['traj_test_n']
results_by_option = data['results_by_option']
res_delayed       = data['res_delayed']
shared_histories  = data['shared_histories']
pa_histories      = data['pa_histories']
delay_labels      = data['delay_labels']
delay_steps_list  = data['delay_steps']

# ── 侧边栏 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    sel_delay = st.selectbox("Communication Delay", delay_labels, index=0)
    vis_len   = st.slider("Visualisation window (frames)", 100, 600, 300, step=50)
    st.markdown("---")
    st.markdown("**Models**")
    for name in models:
        st.markdown(f"- {name}")
    st.markdown("---")
    st.caption("Delay steps: 100ms=3, 200ms=6, 500ms=15 frames @ 30Hz")

delay_idx = delay_labels.index(sel_delay)
ds        = delay_steps_list[delay_idx]   # int，延迟帧数
SEQ_LEN   = 30

# ─────────────────────────────────────────────────────────────────────────────
# Tab 布局
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 RMSE Comparison",
    "📈 3D Trajectory",
    "📉 Training Curves",
    "🔬 Ablation Study",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RMSE 对比表 + 柱状图
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Table I — Prediction RMSE (mm)")

    delay_results = results_by_option['A']
    method_order  = ['Vanilla LSTM', 'Attention-LSTM',
                     'PA-LSTM (w/o PI-Loss)', 'PA-LSTM (Proposed)']

    rows = []
    for lbl in delay_labels:
        row = {'Delay': lbl, 'Delayed (No Comp.)': round(res_delayed[lbl], 4)}
        for name in method_order:
            row[name] = round(delay_results[name][lbl], 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Delay')

    # 高亮最小值
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_min]

    st.dataframe(
        df.style.apply(highlight_min, axis=1),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("RMSE Bar Chart")

    col_left, col_right = st.columns([3, 1])
    with col_right:
        show_delayed = st.checkbox("Show Delayed Baseline", value=True)

    all_methods = (['Delayed (No Comp.)'] if show_delayed else []) + method_order
    colors = {
        'Delayed (No Comp.)': '#EF4444',
        'Vanilla LSTM':       '#60A5FA',
        'Attention-LSTM':     '#34D399',
        'PA-LSTM (w/o PI-Loss)': '#FBBF24',
        'PA-LSTM (Proposed)': '#7C3AED',
    }

    fig_bar = go.Figure()
    for method in all_methods:
        vals = []
        for lbl in delay_labels:
            if method == 'Delayed (No Comp.)':
                vals.append(res_delayed[lbl])
            else:
                vals.append(delay_results[method][lbl])
        fig_bar.add_trace(go.Bar(
            name=method,
            x=delay_labels,
            y=vals,
            marker_color=colors[method],
            text=[f'{v:.4f}' for v in vals],
            textposition='outside',
        ))

    fig_bar.update_layout(
        barmode='group',
        xaxis_title='Communication Delay',
        yaxis_title='RMSE (mm)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450,
        template='plotly_white',
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 3D 轨迹可视化
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"3D Trajectory Comparison — {sel_delay} Latency")

    gt_vis = traj_test_n[SEQ_LEN: SEQ_LEN + vis_len]

    # Delayed baseline
    dl_vis       = np.zeros_like(gt_vis)
    dl_vis[ds:]  = traj_test_n[SEQ_LEN: SEQ_LEN + vis_len - ds]
    dl_vis[:ds]  = traj_test_n[SEQ_LEN]

    # Kalman
    with st.spinner("Computing Kalman Filter..."):
        kf_raw = run_kalman(traj_test_n, ds)
    kf_vis = kf_raw[SEQ_LEN: SEQ_LEN + vis_len]

    # LSTM predictions（滑窗推理，按当前可视化窗口截取）
    with st.spinner("Computing LSTM trajectories..."):
        lstm_colors = {
            'Vanilla LSTM': '#60A5FA',
            'Attention-LSTM': '#34D399',
            'PA-LSTM (w/o PI-Loss)': '#FBBF24',
            'PA-LSTM (Proposed)': '#7C3AED',
        }
        lstm_vis = {}
        for name, mdl in models.items():
            pred_full = run_lstm_trajectory(mdl, traj_test_n, SEQ_LEN, mdl.pred_len)
            lstm_vis[name] = pred_full[SEQ_LEN: SEQ_LEN + vis_len]

    traj_methods = {
        'Ground Truth':           (gt_vis,  '#1E3A5F'),
        f'Delayed {sel_delay}':   (dl_vis,  '#EF4444'),
        'Kalman Filter':          (kf_vis,  '#F59E0B'),
    }
    for name, trj in lstm_vis.items():
        traj_methods[name] = (trj, lstm_colors[name])

    fig3d = go.Figure()
    for label, (trj, color) in traj_methods.items():
        fig3d.add_trace(go.Scatter3d(
            x=trj[:, 0], y=trj[:, 1], z=trj[:, 2],
            mode='lines',
            name=label,
            line=dict(color=color, width=3),
        ))
        # 起点和终点
        fig3d.add_trace(go.Scatter3d(
            x=[trj[0, 0]], y=[trj[0, 1]], z=[trj[0, 2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            showlegend=False,
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title='X (norm)',
            yaxis_title='Y (norm)',
            zaxis_title='Z (norm)',
        ),
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=0, xanchor='right', x=1),
        template='plotly_white',
        title=f'Surgical Trajectory — {sel_delay} Communication Latency',
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # XYZ 分量时序图
    st.markdown("---")
    st.subheader("Per-axis Time Series")
    axis_names = ['X', 'Y', 'Z']
    fig_axes = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=['X axis', 'Y axis', 'Z axis'])
    for ax_i, ax_name in enumerate(axis_names):
        for label, (trj, color) in traj_methods.items():
            fig_axes.add_trace(
                go.Scatter(
                    x=list(range(len(trj))),
                    y=trj[:, ax_i],
                    name=label,
                    line=dict(color=color),
                    showlegend=(ax_i == 0),
                ),
                row=ax_i + 1, col=1
            )
    fig_axes.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig_axes, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 训练曲线
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Training & Validation Loss Curves")

    all_histories = {**shared_histories}
    for name, hist in pa_histories.items():
        # pa_histories 保存的是 (tr_h, va_h) tuple
        all_histories[f'PA-LSTM ({name})'] = hist if isinstance(hist, tuple) else hist

    model_colors = {
        'Vanilla LSTM':       '#60A5FA',
        'Attention-LSTM':     '#34D399',
        'PA-LSTM (w/o PI-Loss)': '#FBBF24',
        'PA-LSTM (Proposed)': '#7C3AED',
    }

    col1, col2 = st.columns(2)
    for col, loss_type, idx in [(col1, 'Train Loss', 0), (col2, 'Val Loss', 1)]:
        with col:
            fig_loss = go.Figure()
            for name, hist_tuple in shared_histories.items():
                tr_h, va_h = hist_tuple
                curve = tr_h if idx == 0 else va_h
                fig_loss.add_trace(go.Scatter(
                    y=curve,
                    name=name,
                    line=dict(color=model_colors.get(name, '#888'), width=2),
                ))
            # PA 历史
            for name, hist_tuple in pa_histories.items():
                tr_h, va_h = hist_tuple
                curve = tr_h if idx == 0 else va_h
                fig_loss.add_trace(go.Scatter(
                    y=curve,
                    name=name,
                    line=dict(
                        color=model_colors.get(name, '#888'),
                        width=2,
                        dash='dash' if 'w/o' in name else 'solid',
                    ),
                ))
            fig_loss.update_layout(
                title=loss_type,
                xaxis_title='Epoch',
                yaxis_title='Loss',
                template='plotly_white',
                height=400,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Ablation Study — Effect of PI-Loss")
    st.markdown(
        "Comparing **PA-LSTM (Proposed)** (with Physics-Informed Loss) "
        "vs **PA-LSTM (w/o PI-Loss)** to quantify the contribution of the physics constraint."
    )

    delay_results = results_by_option['A']
    ablation_rows = []
    for lbl in delay_labels:
        no_pi    = delay_results['PA-LSTM (w/o PI-Loss)'][lbl]
        proposed = delay_results['PA-LSTM (Proposed)'][lbl]
        improve  = (no_pi - proposed) / no_pi * 100
        ablation_rows.append({
            'Delay':                    lbl,
            'PA-LSTM (w/o PI-Loss)':    round(no_pi, 4),
            'PA-LSTM (Proposed)':       round(proposed, 4),
            'Improvement (%)':          round(improve, 2),
        })

    df_ab = pd.DataFrame(ablation_rows).set_index('Delay')
    st.dataframe(df_ab, use_container_width=True)

    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(
        x=delay_labels,
        y=[delay_results['PA-LSTM (w/o PI-Loss)'][l] for l in delay_labels],
        name='PA-LSTM (w/o PI-Loss)',
        mode='lines+markers',
        line=dict(color='#FBBF24', width=3, dash='dash'),
        marker=dict(size=10),
    ))
    fig_ab.add_trace(go.Scatter(
        x=delay_labels,
        y=[delay_results['PA-LSTM (Proposed)'][l] for l in delay_labels],
        name='PA-LSTM (Proposed) ★',
        mode='lines+markers',
        line=dict(color='#7C3AED', width=3),
        marker=dict(size=10, symbol='star'),
    ))
    fig_ab.update_layout(
        xaxis_title='Communication Delay',
        yaxis_title='RMSE (mm)',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown("---")
    st.subheader("All Methods Comparison")
    fig_all = go.Figure()
    all_methods_ab = ['Vanilla LSTM', 'Attention-LSTM',
                      'PA-LSTM (w/o PI-Loss)', 'PA-LSTM (Proposed)']
    ab_colors = ['#60A5FA', '#34D399', '#FBBF24', '#7C3AED']
    ab_dashes  = ['dot', 'dot', 'dash', 'solid']
    for name, color, dash in zip(all_methods_ab, ab_colors, ab_dashes):
        fig_all.add_trace(go.Scatter(
            x=delay_labels,
            y=[delay_results[name][l] for l in delay_labels],
            name=name,
            mode='lines+markers',
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8),
        ))
    fig_all.add_trace(go.Scatter(
        x=delay_labels,
        y=[res_delayed[l] for l in delay_labels],
        name='Delayed (No Comp.)',
        mode='lines+markers',
        line=dict(color='#EF4444', width=2, dash='longdash'),
        marker=dict(size=8),
    ))
    fig_all.update_layout(
        xaxis_title='Communication Delay',
        yaxis_title='RMSE (mm)',
        template='plotly_white',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_all, use_container_width=True)
