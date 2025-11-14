import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    load_json_records,
    prepare_dataframe,
    filter_by_date,
    overview_metrics,
    response_time_stats,
    response_time_samples,
    response_time_samples_by_labels,
    dialogue_metrics,
    sentiment_and_emoji,
    shen_short_stats,
    per_user_stats,
    emoji_package_analysis,
)
from analysis import cluster_gaps_multilevel, cluster_gaps_by_jumps, compute_reply_gaps


st.set_page_config(page_title="微信聊天数据分析", layout="wide")
st.title("微信聊天数据分析与可视化")
st.caption("本地处理，不上传数据。支持 JSON 文件上传或本地路径读取。")

with st.sidebar:
    st.header("数据输入")
    uploaded = st.file_uploader("上传聊天记录 JSON 文件", type=["json","jsonl","txt"])
    path = st.text_input("或指定本地 JSON 文件路径")
    # 明确显示当前选定的文件或路径，避免用户误解来源
    if uploaded is not None:
        try:
            st.caption(f"已选择文件：{getattr(uploaded, 'name', '未知文件名')}")
        except Exception:
            st.caption("已选择文件：未知文件名")
    elif path:
        st.caption(f"已指定路径：{path}")
    else:
        st.caption("未上传/未指定路径时，默认载入示例数据 sample_chat.json")
    st.divider()
    st.header("分析设置")
    max_reply_hours = st.slider("响应时间统计的最大间隔（小时）", 1, 72, 24)
    st.caption("会话分割：自动基于时间间隔的聚类，无需手动设置阈值。")
    st.divider()
    date_range = st.date_input(
        "选择时间范围（可选）",
        value=[],
    )

def load_df():
    records = []
    source = None
    if uploaded is not None:
        records = load_json_records(uploaded)
        if not records:
            st.error("上传的文件解析失败。请确认内容为 JSON/JSONLines 格式，且编码为常见编码（建议 UTF-8）。")
        source = "uploaded"
    elif path:
        records = load_json_records(path)
        source = "path"
    else:
        sample_path = Path(__file__).resolve().parent / "sample_chat.json"
        if sample_path.exists():
            records = load_json_records(str(sample_path))
            source = "sample"
    df = prepare_dataframe(records, me_sender_flag=1)
    # Date filter
    if not df.empty and len(date_range) == 2:
        start = datetime.combine(date_range[0], datetime.min.time())
        end = datetime.combine(date_range[1], datetime.max.time())
        df = filter_by_date(df, start, end)
    return df, source

df, source = load_df()

if df is None or df.empty:
    st.warning("未检测到数据。请上传 JSON 文件或指定本地路径；如无可用，请确认示例文件 sample_chat.json 是否存在。")
    st.stop()

source_map = {"uploaded": "上传数据", "path": "本地路径", "sample": "示例数据"}
st.success(f"已加载 {len(df)} 条消息（来源：{source_map.get(source, '未知')}）。时间范围: {df['CreateTime'].min()} 至 {df['CreateTime'].max()}")

# 参与者命名（读取ID并在标签中展示）
with st.sidebar:
    st.header("参与者命名")
    # 通用识别：扫描常见ID列，提取两个参与者ID
    id_cols = [
        "talker","room_name","fromWxid","from_wxid","toWxid","to_wxid",
        "sender_wxid","receiver_wxid","wxid","id","owner","self_wxid",
        "my_wxid","user_self"
    ]
    series_list = []
    for col in id_cols:
        if col in df.columns:
            try:
                s = df[col].dropna().astype(str)
                s = s[s.str.len() >= 4]
                series_list.append(s)
            except Exception:
                pass
    if series_list:
        combined = pd.concat(series_list, ignore_index=True)
        counts = combined.value_counts()
        ids = counts.index.tolist()[:2]
        id_A = ids[0] if len(ids) >= 1 else None
        id_B = ids[1] if len(ids) >= 2 else None
    else:
        id_A = None
        id_B = None

    # 规范化ID：去除空白，过滤过短值；避免A/B相同导致重复
    def _clean_id_str(v):
        try:
            sv = str(v).strip()
            return sv if len(sv) >= 4 else ""
        except Exception:
            return ""
    id_A = _clean_id_str(id_A)
    id_B = _clean_id_str(id_B)
    if id_A and id_B and id_A == id_B:
        # 当无法区分双方ID时，取消B以避免重复显示
        id_B = ""

    # 命名遵循固定语义：is_sender=1 为“我”，is_sender=0 为“对方”，仅自定义显示名称
    label_me = "我（is_sender=1）显示名称"
    label_you = "对方（is_sender=0）显示名称"
    name_me = st.text_input(label_me, value="我")
    name_you = st.text_input(label_you, value="对方")
    # 展示 is_sender 对应的 talker 值（参考）
    talker_me = ""
    talker_you = ""
    try:
        if "talker" in df.columns:
            s_me = df[df["is_sender"] == 1]["talker"].dropna().astype(str).str.strip()
            s_you = df[df["is_sender"] == 0]["talker"].dropna().astype(str).str.strip()
            def _top_val(series: pd.Series) -> str:
                try:
                    series = series[series.str.len() >= 1]
                    counts = series.value_counts()
                    return str(counts.index[0]) if len(counts) > 0 else ""
                except Exception:
                    return ""
            talker_me = _top_val(s_me)
            talker_you = _top_val(s_you)
    except Exception:
        pass
    st.caption(
        f"检测到的参与者ID（参考）：我(is_sender=1)的talker={talker_me or '未识别'} | 对方(is_sender=0)的talker={talker_you or '未识别'}"
    )

#
# 备注移除：按行级ID构建显示参与者列说明（原为顶部说明文本）。
# 若存在 talker 且与 is_sender 绑定，用它来推断双方ID更稳健
# 直接基于 is_sender 构造显示名称映射，避免 A/B 反转

def _safe_label(s, default):
    try:
        sv = str(s).strip()
        return sv if sv else default
    except Exception:
        return default
name_me = _safe_label(name_me, "我")
name_you = _safe_label(name_you, "对方")
#
# 备注移除：双方输入显示名称相同或为空的说明（原为顶部说明文本）。
try:
    if str(name_me).strip() == str(name_you).strip():
        # 若用户把双方名称设为相同，回退为默认“我/对方”以保证分组与样本计算正确
        name_me = "我"
        name_you = "对方"
except Exception:
    # 任意异常情况下也使用默认区分名称
    name_me = name_me or "我"
    name_you = name_you or "对方"

name_map = {"我": name_me, "对方": name_you}
# 向量化映射显示名，避免逐行 apply 的开销
try:
    sender_str = df["sender"].astype(str)
    df["display_sender"] = sender_str.map(name_map).fillna(sender_str)
except Exception:
    df["display_sender"] = df["sender"].apply(lambda s: name_map.get(str(s), str(s)))

# ------------------------
# 缓存重计算函数，避免每次控件变动导致全量重算
# ------------------------
@st.cache_data(show_spinner=False)
def cached_overview_metrics(df_input: pd.DataFrame):
    return overview_metrics(df_input)

@st.cache_data(show_spinner=False)
def cached_dialogue_metrics(
    df_input: pd.DataFrame,
    sessionize_mode: str = "adaptive_iqr",
    max_reply_hours: int = 24,
    iqr_gap_multiplier: float = 1.5,
    gap_sensitivity: float = 1.8,
    window_min: int = 5,
    window_max: int = 50,
    merge_factor: float = 0.5,
    # tslearn 参数（可选）
    ts_n_clusters: int = 3,
    ts_metric: str = "euclidean",
    ts_coarse_factor: float = 1.5,
):
    return dialogue_metrics(
        df_input,
        sessionize_mode=sessionize_mode,
        max_reply_hours=max_reply_hours,
        iqr_gap_multiplier=iqr_gap_multiplier,
        adaptive_gap_sensitivity=gap_sensitivity,
        adaptive_window_min=int(window_min),
        adaptive_window_max=int(window_max),
        adaptive_merge_factor=merge_factor,
        tslearn_n_clusters=int(ts_n_clusters),
        tslearn_metric=str(ts_metric),
        tslearn_coarse_factor=float(ts_coarse_factor),
    )

@st.cache_data(show_spinner=False)
def cached_compute_reply_gaps(df_input: pd.DataFrame):
    return compute_reply_gaps(df_input)

@st.cache_data(show_spinner=False)
def cached_cluster_gaps_multilevel(gaps_input: list, k: int = 3):
    return cluster_gaps_multilevel(gaps_input, k=k)

@st.cache_data(show_spinner=False)
def cached_cluster_gaps_by_jumps(
    gaps_input: list,
    min_group_size: int = 5,
    ratio_threshold: float = 3.0,
    diff_quantile: float = 0.9,
    target_groups: int | None = 3,
):
    return cluster_gaps_by_jumps(
        gaps_input,
        min_group_size=min_group_size,
        ratio_threshold=ratio_threshold,
        diff_quantile=diff_quantile,
        target_groups=target_groups,
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["总览", "双方对比", "对话模式", "内容分析", "表情包分析", "更新记录"]) 

with tab1:
    st.subheader("总览指标")
    metrics = cached_overview_metrics(df)
    if not metrics:
        st.warning("数据为空或无法计算总览指标。")
    else:
        # Sender share
        c1, c2 = st.columns(2)
        with c1:
            # 改为按ID映射后的显示名统计
            sender_counts_named = df["display_sender"].value_counts().to_dict()
            fig = px.pie(
                names=list(sender_counts_named.keys()),
                values=list(sender_counts_named.values()),
                title="双方消息占比",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
        with c2:
            # 使用原始类型字段进行类型分布
            chosen_field = "type_name"
            type_counts_over = df[chosen_field].value_counts().to_dict()
            fig = px.bar(x=list(type_counts_over.keys()), y=list(type_counts_over.values()), title="消息类型分布（原始类型）")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        # Heatmap (custom): opacity encodes intensity, 0 -> fully transparent; square cells; x-axis with gaps; transparent background
        st.markdown("**总览时间分布热力图**（透明度表示强度；正方形格；横轴留间隔；透明背景）")
        heat = metrics["heatmap_pivot"].copy()
        for col in range(7):
            if col not in heat.columns:
                heat[col] = 0
        heat = heat[[0,1,2,3,4,5,6]]
        heat.index.name = "月份-周次"
        
        n_rows = len(heat.index)
        n_cols = 7
        base_color = "#1f77b4"
        cell_w = 1.0
        cell_h = 1.0  # square cells
        gap_x = 0.2  # horizontal gap between days
        corner_r = 0.0  # no rounding for squares

        # Normalize opacities
        max_val = float(heat.values.max()) if heat.values.size else 1.0
        def opacity_for(v):
            if v <= 0 or max_val <= 0:
                return 0.0
            return float(v) / max_val

        fig = go.Figure()
        # Draw square cells as rectangles
        for i, row_label in enumerate(heat.index.tolist()):
            for j in range(n_cols):
                v = float(heat.iloc[i, j])
                # x with gaps
                x0 = j * (cell_w + gap_x)
                x1 = x0 + cell_w
                # y (top-down order), we'll reverse y-axis later
                y0 = i
                y1 = i + cell_h
                fig.add_shape(
                    type="rect",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    fillcolor=base_color,
                    line=dict(width=0),
                    opacity=opacity_for(v),
                )

        # Axis ticks
        x_centers = [j * (cell_w + gap_x) + cell_w/2.0 for j in range(n_cols)]
        x_labels = ["周一","周二","周三","周四","周五","周六","周日"]
        y_centers = [i + cell_h/2.0 for i in range(n_rows)]
        y_labels = heat.index.tolist()

        fig.update_xaxes(
            range=[-gap_x/2.0, n_cols * (cell_w + gap_x) - gap_x/2.0],
            tickvals=x_centers,
            ticktext=x_labels,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            range=[0, n_rows],
            tickvals=y_centers,
            ticktext=y_labels,
            showgrid=False,
            zeroline=False,
            autorange="reversed",  # top row first
            scaleanchor="x",
            scaleratio=1,
        )
        # 透明 Heatmap 用于悬浮显示具体日期与消息数（不改变现有透明度视觉样式）
        date_map = df.groupby(["month_week_label", "weekday"])['date'].min()
        row_labels = heat.index.tolist()
        date_matrix = []
        for rl in row_labels:
            row_dates = []
            for j in range(n_cols):
                dval = date_map.get((rl, j))
                row_dates.append(str(dval) if pd.notna(dval) else "")
            date_matrix.append(row_dates)

        heat_hover = go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=heat.values,
            customdata=date_matrix,
            showscale=False,
            opacity=0.001,
            hovertemplate="日期=%{customdata}<br>消息数=%{z}<extra></extra>",
        )
        fig.add_trace(heat_hover)
        fig.update_layout(
            title="总览时间分布热力图",
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        # Day-of-week most & time-of-day avg
        dow_counts = metrics["dow_counts"]
        dow_names = ["周一","周二","周三","周四","周五","周六","周日"]
        dow_values = [dow_counts.get(i, 0) for i in range(7)]
        fig = px.bar(x=dow_names, y=dow_values, title="周几聊得最多（总量）")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        avg_slot = metrics["avg_slot"]
        slot_labels = metrics.get("slot_labels", [f"{i:02d}:00-{(i+1)%24:02d}:00" for i in range(24)])
        # 图形类型切换（圆环图/柱状图）
        view_choice = st.radio("图形类型", ["圆环图", "柱状图"], index=0, horizontal=True)

        # 预计算与颜色透明度映射
        theta_deg = [i * (360/24) for i in range(24)]
        width_deg = [360/24] * 24
        tick_text = [f"{i:02d}:00" for i in range(24)]
        base_rgb = (31, 119, 180)  # #1f77b4
        vals = list(avg_slot.values)
        if len(vals) == 0:
            vals = [0] * 24
        vmin = float(min(vals))
        vmax = float(max(vals))
        vmax_safe = vmax if vmax > 0 else 1.0
        colors = []
        for v in vals:
            if vmax == vmin:
                alpha = 0.6
            else:
                t = (float(v) - vmin) / (vmax - vmin)
                alpha = 0.15 + 0.85 * t
            colors.append(f"rgba({base_rgb[0]},{base_rgb[1]},{base_rgb[2]},{alpha})")

        if view_choice == "柱状图":
            fig_bar = px.bar(x=slot_labels, y=vals, title="聊天时段分布（24段，平均每日）", labels={"x":"时段","y":"平均消息数"})
            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["zoomIn2d","zoomOut2d"], "toImageButtonOptions": {"format": "svg"}})
        else:
            fig_ring = go.Figure()
            fig_ring.add_trace(go.Barpolar(
                r=vals,
                theta=theta_deg,
                width=width_deg,
                marker=dict(color=colors),
                hovertemplate="时段=%{text}<br>平均消息数=%{r}<extra></extra>",
                text=slot_labels,
                name="时段"
            ))
            fig_ring.update_layout(
                title="聊天时段分布（24段圆环，平均每日）",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(
                        direction="clockwise",
                        rotation=90,
                        tickmode="array",
                        tickvals=theta_deg,
                        ticktext=tick_text,
                    ),
                    radialaxis=dict(visible=False, range=[0, vmax_safe])
                )
            )
            st.plotly_chart(fig_ring, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

with tab2:
    st.subheader("双方聊天特点对比")
    stats = per_user_stats(df, sender_col="display_sender")
    # 响应时间样本计算：仅使用 talker 作为唯一ID，不再回退到 sender
    # 修复：响应时间采样应按每条消息的 sender（“我”/“对方”）来统计，而不是 talker/chat id
    samples_named = response_time_samples_by_labels(
        df,
        "我",
        "对方",
        sender_col="sender",
        max_reply_hours=max_reply_hours,
    )
    if stats:
        c1, c2 = st.columns(2)
        with c1:
            # 平均消息长度（按用户算法）：严格仅统计原始类型为“文本”的消息，并按每条消息的 sender（“我”/“对方”）分组
            text_df = df[df["type_name"] == "文本"].copy()
            def _fmt_num_or_na(x):
                try:
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return "N/A"
                    return f"{float(x):.1f}"
                except Exception:
                    return "N/A"
            def _mean_len_by_sender(sender_label):
                try:
                    sub = text_df[text_df["sender"].astype(str) == str(sender_label)]
                    if len(sub) == 0:
                        return float('nan'), 0
                    lengths = pd.to_numeric(sub["message_length"], errors="coerce")
                    total_len = float(np.nansum(lengths.values))
                    cnt = int(len(sub))
                    avg = (total_len / cnt) if cnt > 0 else float('nan')
                    return float(avg), cnt
                except Exception:
                    return float('nan'), 0
            # 按 sender 的“我/对方”分组统计
            me_val, me_cnt = _mean_len_by_sender("我")
            you_val, you_cnt = _mean_len_by_sender("对方")
            me_label = name_map.get('我')
            you_label = name_map.get('对方')
            st.metric(f"{me_label}-平均消息长度", _fmt_num_or_na(me_val))
            st.caption(f"文本样本数: {me_cnt}")
            st.metric(f"{you_label}-平均消息长度", _fmt_num_or_na(you_val))
            st.caption(f"文本样本数: {you_cnt}")
        with c2:
            # 按显示名分组的类型分布（使用原始类型字段）
            chosen_field = "type_name"
            type_dist = df.groupby(["display_sender", chosen_field]).size().reset_index(name="count")
            type_pivot2 = type_dist.pivot(index=chosen_field, columns="display_sender", values="count").fillna(0)
            fig = px.bar(type_pivot2, barmode="group", title="每人消息类型分布（原始类型）")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
    if samples_named:
        c3, c4 = st.columns(2)
        with c3:
            s_me = samples_named.get("a_to_b", [])
            def _fmt_time_or_na(arr):
                try:
                    if not arr or len(arr) == 0:
                        return "N/A"
                    return f"{float(np.median(arr)):.1f}"
                except Exception:
                    return "N/A"
            st.metric(f"{name_map.get('我')}→{name_map.get('对方')} 响应时间中位数(秒)", _fmt_time_or_na(s_me))
            st.caption(f"样本数: {len(s_me)}")
        with c4:
            s_you = samples_named.get("b_to_a", [])
            st.metric(f"{name_map.get('对方')}→{name_map.get('我')} 响应时间中位数(秒)", _fmt_time_or_na(s_you))
            st.caption(f"样本数: {len(s_you)}")

        st.markdown("**响应时间分布（直方图）**")
        bins = st.slider("分箱数", 10, 100, 30, step=5)
        logt = st.checkbox("对时间值取自然对数（ln）", value=False)
        logy = st.checkbox("对频数也取自然对数（ln）", value=False)
        samples = samples_named

        c5, c6 = st.columns(2)
        with c5:
            s_me = samples.get("a_to_b", [])
            s_me_plot = ([np.log(float(v)) for v in s_me if (v is not None and float(v) > 0)] if logt else s_me)
            x_label = "ln(秒)" if logt else "秒"
            y_label = "ln(频数)" if logy else "频数"
            # 手动统计分箱并对频数可选取ln
            if len(s_me_plot) > 0:
                data_arr = np.asarray(s_me_plot, dtype=float)
                counts, edges = np.histogram(data_arr, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2.0
                if logy:
                    mask = counts > 0
                    x_vals = centers[mask]
                    y_vals = np.log(counts[mask])
                else:
                    x_vals = centers
                    y_vals = counts
                fig_me = px.bar(x=x_vals, y=y_vals, title=f"{name_map.get('我')}→{name_map.get('对方')} 响应时间分布", labels={"x": x_label, "y": y_label})
            else:
                fig_me = px.histogram(x=s_me_plot, nbins=bins, title=f"{name_map.get('我')}→{name_map.get('对方')} 响应时间分布", labels={"x": x_label, "y": y_label})
            st.plotly_chart(fig_me, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
            if s_me:
                q = np.percentile(s_me, [25, 50, 75])
                st.caption(f"分位数 Q1={q[0]:.1f}s, Q2(中位)={q[1]:.1f}s, Q3={q[2]:.1f}s")
        with c6:
            s_you = samples.get("b_to_a", [])
            s_you_plot = ([np.log(float(v)) for v in s_you if (v is not None and float(v) > 0)] if logt else s_you)
            x_label2 = "ln(秒)" if logt else "秒"
            y_label2 = "ln(频数)" if logy else "频数"
            if len(s_you_plot) > 0:
                data_arr2 = np.asarray(s_you_plot, dtype=float)
                counts2, edges2 = np.histogram(data_arr2, bins=bins)
                centers2 = (edges2[:-1] + edges2[1:]) / 2.0
                if logy:
                    mask2 = counts2 > 0
                    x_vals2 = centers2[mask2]
                    y_vals2 = np.log(counts2[mask2])
                else:
                    x_vals2 = centers2
                    y_vals2 = counts2
                fig_you = px.bar(x=x_vals2, y=y_vals2, title=f"{name_map.get('对方')}→{name_map.get('我')} 响应时间分布", labels={"x": x_label2, "y": y_label2})
            else:
                fig_you = px.histogram(x=s_you_plot, nbins=bins, title=f"{name_map.get('对方')}→{name_map.get('我')} 响应时间分布", labels={"x": x_label2, "y": y_label2})
            st.plotly_chart(fig_you, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
            if s_you:
                q2 = np.percentile(s_you, [25, 50, 75])
                st.caption(f"分位数 Q1={q2[0]:.1f}s, Q2(中位)={q2[1]:.1f}s, Q3={q2[2]:.1f}s")

with tab3:
    st.subheader("对话模式分析（自适应间隔）")

    # 算法与指标说明（在分析前展示）
    with st.expander("算法与指标说明", expanded=True):
        st.markdown(
            """
            算法直观解释（自适应间隔）
            - 把消息看作一条时间线。我们观察相邻消息的时间间隔 `Δt`，用一个“滑动窗口”在局部计算这段时间的“常态速度”。
            - 常态用两个数字刻画：局部中位数 `local_median`（典型间隔）和局部标准差 `local_std`（波动程度）。
            - 若当前间隔 `Δt_i` 超过“常态阈值”`local_median + k × local_std`（k=敏感度），说明这次停顿明显变长 → 认定这里是一个“对话分界点”。
            - 于是把上一段会话关掉（到第 i 条消息），并从下一条消息 i+1 开始新会话；尾部自然收尾到最后一条消息。
            - 相邻会话之间如果几乎没有空隙（间隙 < 全局中位间隔 × 合并系数），就把它们合并为一个会话，避免切得过碎。
            - 可视化约定：绿色虚线是“会话中心”（消息时间中位数，更贴近密集区），默认展示全部散点并带悬停信息。

            指标怎么读
            - 全局中位数/均值（秒）：数据整体的“典型间隔/平均间隔”，可帮助判断合并阈值的量级。
            - 会话数 / 双人互动 / 单人独白：切分结果的规模与类型分布。
            - 段明细表：每段的起止、持续时长、消息数、类型，便于核查切分是否合理。

            调参小抄（常用场景）
            - 会话过碎：增大 `k`（比如 1.8→2.2）或增大 `merge_factor`（比如 0.5→0.8）。
            - 会话过长：减小 `k`（比如 1.8→1.4）或减小 `merge_factor`（比如 0.5→0.3）。
            - 对话密度变化很快：减小 `window_min`/`window_max`（更敏捷），如 `5/50→4/30`；密度变化很慢则可增大（更稳健）。
            """
        )

    # 参数调节（仅对话模式页）
    with st.expander("切分参数设置", expanded=True):
        c_left, c_right = st.columns([2, 1])
        with c_left:
            k = st.slider("敏感度 k（gap_sensitivity）", 1.0, 3.0, 1.8, step=0.1)
            window_min = st.number_input("滑动窗口最小长度（window_min）", min_value=3, max_value=100, value=5, step=1)
            window_max = st.number_input("滑动窗口最大长度（window_max）", min_value=int(window_min), max_value=200, value=50, step=1)
            merge_factor = st.slider("合并系数（merge_factor）", 0.1, 2.0, 0.5, step=0.1)
            st.caption("阈值：Δt_i > local_median + k × local_std → 分界；合并：相邻段空隙 < 全局中位间隔 × merge_factor")
        with c_right:
            st.markdown(
                """
                参数提示
                - k↑：更宽松（段更长）
                - k↓：更严格（段更短）
                - window：变小更敏捷，变大更稳健
                - 合并系数↑：更容易合并
                """
            )
        # 提供刷新缓存按钮，避免旧算法结果残留
        col_btn = st.columns(1)[0]
        with col_btn:
            if st.button("刷新分段缓存并重算", use_container_width=True):
                try:
                    cached_dialogue_metrics.clear()
                    st.success("已清除缓存，下一次请求将重新计算。")
                except Exception:
                    st.info("缓存清除失败或无缓存。请稍后重试。")
    # 已移除 tslearn 算法参数面板

    # 使用 IQR 自适应算法进行会话分割
    session_mode = "adaptive_iqr"
    dia = cached_dialogue_metrics(
        df,
        sessionize_mode=session_mode,
        max_reply_hours=max_reply_hours,
        iqr_gap_multiplier=1.0,
        gap_sensitivity=k,
        window_min=int(window_min),
        window_max=int(window_max),
        merge_factor=merge_factor,
        # 不再传递 tslearn 参数
    )
    if dia:
        # Show clustering info
        ci = dia.get("cluster_info", {})
        actual_status = str(ci.get("status", "")).lower()
        # 会话统计（先展示“所有会话数”，再细分两人互动与单人独白）
        st.markdown("**会话统计**")
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("所有会话数", ci.get("session_count", 0))
        with cB:
            types = ci.get("session_types", [])
            duo_cnt = int(sum(1 for t in types if t == "双人互动"))
            st.metric("双人互动会话数", duo_cnt)
        with cC:
            mono_cnt = int(sum(1 for t in types if t in ("单人独白段", "单人独白")))
            st.metric("单人独白会话数", mono_cnt)
        # 根据实际算法状态展示对应参数；当 tslearn 不可用或出错时提示并回退展示滑窗指标
        if actual_status == "adaptive_iqr":
            cX, cY, cZ = st.columns(3)
            with cX:
                st.metric("全局中位数/均值(秒)", f"{ci.get('global_median_gap_sec', float('nan')):.1f} / {ci.get('global_mean_gap_sec', float('nan')):.1f}")
            with cY:
                st.metric("敏感度 k", f"{ci.get('gap_sensitivity', k):.2f}")
            with cZ:
                st.metric("窗口范围/合并系数", f"{ci.get('window_min', int(window_min))}-{ci.get('window_max', int(window_max))} / {ci.get('merge_factor', merge_factor):.2f}")
            st.caption("读法：中位/均值给出大致量级；k 控制分界敏感度；窗口控制算法反应速度与稳健性；合并系数控制邻段是否并入。")
        elif actual_status == "tslearn_removed":
            st.info("tslearn 切分已移除，当前使用自适应滑窗算法。")
        else:
            # tslearn 不可用或出错（或其他模式未知），提示并回退显示滑窗指标
            if actual_status in ("tslearn_unavailable", "tslearn_fallback", "tslearn_empty"):
                err = ci.get("error", "")
                st.warning(f"tslearn 不可用或执行出错，已回退到自适应滑窗算法。{('原因：' + str(err)) if err else ''}")
            cX, cY, cZ = st.columns(3)
            with cX:
                st.metric("全局中位数/均值(秒)", f"{ci.get('global_median_gap_sec', float('nan')):.1f} / {ci.get('global_mean_gap_sec', float('nan')):.1f}")
            with cY:
                st.metric("敏感度 k", f"{ci.get('gap_sensitivity', k):.2f}")
            with cZ:
                st.metric("窗口范围/合并系数", f"{ci.get('window_min', int(window_min))}-{ci.get('window_max', int(window_max))} / {ci.get('merge_factor', merge_factor):.2f}")
            st.caption("算法：Δt_i > local_median + k × local_std 判定分界；窗口大小自适应于密度；相邻段空隙 < 全局中位间隔 × merge_factor 自动合并。")

        # 已移除单口相声统计展示

        def fmt_secs(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            try:
                x = float(x)
            except Exception:
                return str(x)
            if x < 60:
                return f"{x:.1f} 秒"
            if x < 3600:
                return f"{x/60:.1f} 分钟"
            return f"{x/3600:.2f} 小时"

        # 时间轴散点：默认展示所有点，并恢复悬浮信息（说话人id、消息类型、内容、时间点）
        timeline_df = df[["CreateTime", "display_sender", "sender", "type_name", "msg"]].copy()
        timeline_df["y"] = 0
        fig_time = go.Figure()
        for name, sub in timeline_df.groupby("display_sender"):
            try:
                custom = np.stack([
                    sub["sender"].astype(str).values,
                    sub["type_name"].astype(str).values,
                    sub["msg"].astype(str).values,
                ], axis=1)
            except Exception:
                # 回退：若 stack 失败，则仅提供 sender 与 type_name
                custom = np.stack([
                    sub["sender"].astype(str).values,
                    sub["type_name"].astype(str).values,
                    np.array([""] * len(sub)),
                ], axis=1)
            fig_time.add_trace(
                go.Scattergl(
                    x=sub["CreateTime"],
                    y=sub["y"],
                    mode="markers",
                    name=str(name),
                    marker=dict(size=4, opacity=0.6),
                    customdata=custom,
                    hovertemplate="说话人ID: %{customdata[0]}<br>类型: %{customdata[1]}<br>内容: %{customdata[2]}<br>时间: %{x}<extra></extra>",
                )
            )
        fig_time.update_layout(title="时间轴散点：消息分布（全部点）")
        fig_time.update_yaxes(visible=False, showticklabels=False)
        # 参考线：仅会话中心（移除灰色边界）
        centers = ci.get("cluster_centers", [])
        for ct in centers:
            try:
                fig_time.add_vline(x=ct, line_dash="dot", line_color="green", opacity=0.6)
            except Exception:
                pass
        st.caption("绿色虚线=会话中心（消息时间中位数）；已移除灰色边界线")
        st.plotly_chart(fig_time, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        # 段明细表：起止时间、持续时长、段类型、消息数（使用 conversations 源数据）
        convs = dia.get("conversations", [])
        if convs:
            try:
                seg_rows = []
                for idx, c in enumerate(convs):
                    s = pd.Timestamp(c.get("start_time"))
                    e = pd.Timestamp(c.get("end_time"))
                    dur_min = (e - s).total_seconds() / 60.0
                    seg_rows.append({
                        "开始": s,
                        "结束": e,
                        "持续(分钟)": round(dur_min, 2),
                        "消息数": int(c.get("message_count", 0)),
                        "类型": (types[idx] if idx < len(types) else ""),
                    })
                st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)
            except Exception:
                pass

        # 回复间隔分布：改为直方图（更快）
        gaps = cached_compute_reply_gaps(df)
        fig_gap = px.histogram(x=gaps, nbins=80, title="回复间隔分布（直方图，快速）", labels={"x": "间隔（秒）", "y": "频次"})
        st.plotly_chart(fig_gap, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        # 回复间隔分类（短/中/长）散点时间轴
        if len(gaps) >= 3:
            cl = cached_cluster_gaps_multilevel(gaps, k=3)
            centers = cl.get("centers_sec", [])
            labels = cl.get("labels", [])
            # 回退：若聚类不稳定或簇数不足，则使用“跳跃边界分组”
            if not centers or len(set(labels)) < 2:
                cl2 = cached_cluster_gaps_by_jumps(
                    gaps,
                    min_group_size=5,
                    ratio_threshold=3.0,
                    diff_quantile=0.9,
                    target_groups=3,
                )
                centers = cl2.get("group_centers_sec", [])
                labels = cl2.get("labels", [])

            # 将簇索引按中心值升序映射为 0/1/2 → 短/中/长
            uniq = sorted([(i, c) for i, c in enumerate(centers)], key=lambda x: x[1])
            rank_map = {idx: rank for rank, (idx, _) in enumerate(uniq)}
            rank_labels = [rank_map.get(l, 0) for l in labels]
            cat_names = ["短", "中", "长"]
            cat_labels = [cat_names[min(r, len(cat_names)-1)] for r in rank_labels]

            # 构造每个间隔对应的时间点（取后一条消息的时间）；避免大DataFrame与无用列以提速
            times_arr = df["CreateTime"].iloc[1:].values
            gaps_arr = np.array(gaps, dtype=float)
            cat_arr = np.array(cat_labels)

            # 分类散点时间轴（WebGL 加速；按类别分别绘制；恢复悬浮信息）
            fig_gap_scatter = go.Figure()
            for cat in ["短", "中", "长"]:
                mask = (cat_arr == cat)
                x_cat = times_arr[mask]
                y_cat = gaps_arr[mask]
                n_cat = len(x_cat)
                if n_cat == 0:
                    continue
                # 构造悬浮信息：说话人id、类型、内容（对应后一条消息）
                sub_df = df.iloc[1:].copy()
                sender_ids = sub_df.loc[mask, "sender"].astype(str).values
                type_vals = sub_df.loc[mask, "type_name"].astype(str).values
                msg_vals = sub_df.loc[mask, "msg"].astype(str).values
                try:
                    custom = np.stack([sender_ids, type_vals, msg_vals], axis=1)
                except Exception:
                    custom = np.stack([sender_ids, type_vals, np.array([""] * len(sender_ids))], axis=1)
                fig_gap_scatter.add_trace(
                    go.Scattergl(
                        x=x_cat,
                        y=y_cat,
                        mode="markers",
                        name=cat,
                        marker=dict(size=4, opacity=0.6),
                        customdata=custom,
                        hovertemplate="类别: %{% if False %}{% endif %}" + cat + "<br>间隔: %{y:.1f}s<br>说话人ID: %{customdata[0]}<br>类型: %{customdata[1]}<br>内容: %{customdata[2]}<br>时间: %{x}<extra></extra>",
                    )
                )
            fig_gap_scatter.update_layout(title="回复间隔分类（散点时间轴，WebGL 加速）")
            fig_gap_scatter.update_yaxes(title_text="间隔（秒）")
            st.plotly_chart(fig_gap_scatter, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

            centers_sorted = [c for _, c in uniq]
            try:
                center_text = ", ".join(f"{float(c):.1f}" for c in centers_sorted)
            except Exception:
                center_text = ", ".join(str(c) for c in centers_sorted)
            st.caption(f"间隔中心值(秒)：{center_text}；分类基于 log1p(k-means)/跳跃边界回退，自动判定短/中/长。")

            # 计算阈值范围：按中心值升序，取相邻中心的中点作为边界
            if len(centers_sorted) >= 2:
                try:
                    centers_sorted = [float(c) for c in centers_sorted]
                except Exception:
                    pass
                if len(centers_sorted) >= 3:
                    thr1 = (centers_sorted[0] + centers_sorted[1]) / 2.0
                    thr2 = (centers_sorted[1] + centers_sorted[2]) / 2.0
                    st.caption(f"短间隔：0～{thr1:.1f} 秒；中间隔：{thr1:.1f}～{thr2:.1f} 秒；长间隔：≥{thr2:.1f} 秒。")
                    st.caption("阈值说明：先对回复间隔进行 log1p 变换并用 k-means 聚3类，按中心值排序后用相邻中心的中点作为分割；当聚类不稳定时回退为基于间隔跳跃的分组。")
                else:
                    thr1 = (centers_sorted[0] + centers_sorted[1]) / 2.0
                    st.caption(f"短间隔：0～{thr1:.1f} 秒；长间隔：≥{thr1:.1f} 秒。")
                    st.caption("阈值说明：按两类中心值的中点作为分割；当聚类不稳定时回退为基于间隔跳跃的分组。")
        else:
            st.info("样本不足，无法进行短/中/长间隔分类。")

        init_counts = dia["initiator_counts"]
        init_counts_named = {name_map.get(k, k): v for k, v in init_counts.items()}
        fig = px.pie(names=list(init_counts_named.keys()), values=list(init_counts_named.values()), title="一般谁先发起对话")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
        # 自适应间隔注释：补充算法与回退逻辑
        st.caption("自适应间隔注释：阈值≈局部中位数×系数，取 max(组内最大间隔, median×系数)；当 Δt>2×阈值时强制切分；切分需前/后间隔均大于段内最大间隔。")
        c1 = st.columns(1)[0]
        with c1:
            st.metric("会话内消息平均间隔(秒)", f"{dia['avg_intra_gap_sec']:.1f}")

        c1, c2 = st.columns(2)
        with c1:
            # 改为直接基于ID映射后的显示名统计链接消息
            link_counts_named = df[df["is_link"]]["display_sender"].value_counts().to_dict()
            fig = px.bar(x=list(link_counts_named.keys()), y=list(link_counts_named.values()), title="谁分享链接更多")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
        with c2:
            probs = dia.get("link_swift_reply_prob", {})
            swift_thr = dia.get("swift_threshold_sec", float('nan'))
            st.metric(f"{name_map.get('我')}分享链接后被迅速回复的概率", f"{(probs.get('me',{}).get('prob', float('nan'))*100 if not np.isnan(probs.get('me',{}).get('prob', float('nan'))) else float('nan')):.1f}%")
            st.caption(f"样本数: {probs.get('me',{}).get('n', 0)} | 迅速阈值≈ {swift_thr:.1f}s")
            st.metric(f"{name_map.get('对方')}分享链接后被迅速回复的概率", f"{(probs.get('you',{}).get('prob', float('nan'))*100 if not np.isnan(probs.get('you',{}).get('prob', float('nan'))) else float('nan')):.1f}%")
            st.caption(f"样本数: {probs.get('you',{}).get('n', 0)} | 迅速阈值≈ {swift_thr:.1f}s")

with tab4:
    st.subheader("内容分析")
    with st.expander("性能选项", expanded=True):
        enable_content = st.checkbox("启用内容分析（SnowNLP/jieba，较慢）", value=False)
    if enable_content:
        se = sentiment_and_emoji(df)
        if se:
            sent = se["sentiment"]
            st.metric("中文情感均值(0-1)", f"{sent['avg']:.3f}")
            st.metric("中文情感中位数(0-1)", f"{sent['median']:.3f}")
            st.caption(f"文本样本数: {sent['n']}")
            st.caption("说明：情感分值来自 SnowNLP，对中文文本打分，范围[0,1]；越接近1表示越积极，越接近0表示越消极。该分值仅用于粗略情绪倾向参考，不等同心理评估。")

            c1, c2 = st.columns(2)
            with c1:
                top_words = se["top_words"]
                if top_words:
                    word_df = pd.DataFrame(top_words, columns=["词语", "频次"]).head(30)
                    fig = px.bar(word_df, x="词语", y="频次", title="高频词语(Top 30)")
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
                else:
                    st.info("未能提取词频（可能未安装 jieba 或文本为空）。")
            with c2:
                top_emojis = se["top_emojis"]
                if top_emojis:
                    em_df = pd.DataFrame(top_emojis, columns=["Emoji", "频次"]).head(30)
                    fig = px.bar(em_df, x="Emoji", y="频次", title="表情/Emoji 使用频次(Top 30)")
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
                else:
                    st.info("未检测到 Emoji（文本为空或未安装 emoji 库）。")
    else:
        st.info("为提升整体速度，内容分析默认关闭。勾选上方开关以启用计算。")

        # “神”短文本统计（≤3字且含“神”）
        st.subheader("“神”短文本统计（≤3字且含“神”）")
        shen = shen_short_stats(df)
        if shen:
            st.metric("“神”短文本条数", shen.get("count", 0))
            recs = shen.get("records", [])
            if recs:
                vis_df = pd.DataFrame(recs)
                # 显示者名称替换为侧边栏命名（如必要）
                vis_df["sender"] = vis_df["sender"].apply(lambda s: name_map.get(s, s))
                fig_shen = px.scatter(
                    vis_df,
                    x="time",
                    y=[0] * len(vis_df),
                    title="“神”短文本出现时间点",
                    hover_data={"sender": True, "date": True, "text": True},
                )
                fig_shen.update_yaxes(visible=False, showticklabels=False)
                st.plotly_chart(fig_shen, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
            else:
                st.info("未检出符合条件的“神”短文本。")
        else:
            st.info("数据为空或无法计算“神”统计。")

st.caption("说明：响应时间等统计做了异常值裁剪，具体阈值可在侧边栏调节。")

with tab5:
    st.subheader("表情包分析（type_name=动画表情）")

    with st.expander("分割参数设置（自适应滑窗，与对话模式一致）", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            emoji_k = st.slider("敏感度 k（gap_sensitivity）", 1.0, 3.0, 1.8, step=0.1, key="emoji_k")
            emoji_merge_factor = st.slider("合并系数（merge_factor）", 0.1, 2.0, 0.5, step=0.1, key="emoji_merge")
        with c2:
            emoji_window_min = st.number_input("滑动窗口最小长度（window_min）", min_value=3, max_value=100, value=5, step=1, key="emoji_wmin")
            emoji_window_max = st.number_input("滑动窗口最大长度（window_max）", min_value=int(emoji_window_min), max_value=200, value=50, step=1, key="emoji_wmax")
        st.caption("分界：Δt_i > local_median + k × local_std；窗口大小随密度自适应；合并：相邻段空隙 < 全局中位间隔 × merge_factor 时合并。")

    # 对话切分与“对话模式”保持一致：使用 adaptive_iqr（自适应滑窗）
    em = emoji_package_analysis(
        df,
        sessionize_mode="adaptive_iqr",
        iqr_gap_multiplier=1.0,
        gap_sensitivity=emoji_k,
        window_min=int(emoji_window_min),
        window_max=int(emoji_window_max),
        merge_factor=emoji_merge_factor,
    )

    if em:
        # 1) 每轮会话表情包占比
        props = em["session_props"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("会话数", props.get("n_sessions", 0))
        with c2:
            st.metric("会话表情包占比均值", f"{props.get('mean', float('nan')):.3f}")
        with c3:
            st.metric("会话表情包占比中位数", f"{props.get('median', float('nan')):.3f}")

        vals = props.get("values", [])
        # 会话占比分布（柱状图）
        prop_df = pd.DataFrame({"会话索引": list(range(1, len(vals)+1)), "占比": vals})
        fig_props = px.bar(prop_df, x="会话索引", y="占比", title="每轮会话表情包使用占比")
        st.plotly_chart(fig_props, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

        # 2) 一条消息后接表情包的概率
        npb = em["next_probs"]
        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("整体概率", f"{(npb.get('overall', float('nan'))*100):.1f}%")
        with c5:
            st.metric(f"{name_map.get('我')}发后下一条是表情包概率", f"{(npb.get('me', float('nan'))*100):.1f}%")
        with c6:
            st.metric(f"{name_map.get('对方')}发后下一条是表情包概率", f"{(npb.get('you', float('nan'))*100):.1f}%")
        st.caption(f"样本对数：整体 {npb['counts']['pairs']}，我 {npb['counts']['me_pairs']}，对方 {npb['counts']['you_pairs']}")

        # 3) 连续>=3表情包的频率
        seq = em["seq_stats"]
        st.metric("连续>=3表情包出现次数", seq.get("count", 0))
        st.metric("连续>=3表情包每百条消息出现次数", f"{seq.get('rate_per_100_msgs', float('nan')):.2f}")
        st.caption("注：按所有消息顺序扫描，统计连续动画表情段长度≥3的出现次数。")

        # 会话切分参数回显（与对话模式一致）
        ci2 = em.get("sessionize_info", {})
        c7, c8, c9 = st.columns(3)
        with c7:
            st.metric("敏感度 k", f"{ci2.get('gap_sensitivity', emoji_k):.2f}")
        with c8:
            st.metric("窗口范围", f"{ci2.get('window_min', int(emoji_window_min))}-{ci2.get('window_max', int(emoji_window_max))}")
        with c9:
            st.metric("合并系数", f"{ci2.get('merge_factor', emoji_merge_factor):.2f}")

        # 4) 表情包内容(src)频次排行
        top_pkgs = em.get("package_top", [])
        if top_pkgs:
            st.subheader("最常用的表情包（按src内容统计）")
            pkg_df = pd.DataFrame(top_pkgs, columns=["src", "频次"])[:20]
            # 为显示友好，截断过长的src
            pkg_df["src_预览"] = pkg_df["src"].apply(lambda s: (s if len(str(s)) <= 80 else str(s)[:77] + "...") )
            fig_pkg = px.bar(pkg_df, x="src_预览", y="频次", title="表情包频次排行（Top 20）")
            fig_pkg.update_layout(xaxis_title="src(截断显示)", yaxis_title="频次")
            st.plotly_chart(fig_pkg, use_container_width=True, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})

            # 可跳转链接按钮（Top 12 可点击的 src）
            clickable = [(str(src), int(cnt)) for src, cnt in top_pkgs if isinstance(src, str) and str(src).lower().startswith(("http://", "https://"))]
            if clickable:
                st.subheader("快速访问表情包内容（Top 链接）")
                cols = st.columns(3)
                for i, (url, cnt) in enumerate(clickable[:12]):
                    btn_label = f"打开({cnt}次)"
                    with cols[i % 3]:
                        try:
                            # Streamlit >=1.25 支持 link_button
                            st.link_button(btn_label, url)
                        except Exception:
                            st.markdown(f"[{btn_label}]({url})")
                        # 显示截断预览与完整链接提示
                        st.caption((url if len(url) <= 80 else (url[:77] + "...")))
            else:
                st.info("Top src 中未发现可直接访问的链接内容。")
        else:
            st.info("未检测到表情包src内容或数据为空。")
    else:
        st.info("未检测到动画表情(type_name=动画表情)消息，暂不展示统计。")

with tab6:
    st.subheader("更新记录")
    st.caption("按版本记录近期更新，格式示例：x.xx.x。")
    st.markdown(
        """
        **语义化版本说明**：第二位为功能更新（Minor）；第三位为修复或非功能性改进（Patch）。

        **1.4.2**
        - 清除 tslearn 算法分段实现与参数面板，统一回退为“自适应滑窗”，不影响其他已有功能与可视化。
        
        **1.4.1**
        - 废除ruptures方案，更新 tslearn 方案。

        **1.4.0**
        - 更新 ruptures 方案。

        **1.3.3**
        - 移除灰色会话边界虚线，仅保留绿色会话中心参考线，降低绘制开销。
        - 恢复散点悬浮信息（说话人ID、消息类型、内容、时间点），默认展示所有散点。
        - 取消密度热力图与可视化性能选项，避免分散关注与提升交互流畅度。
        - 新增 IQR 倍数解释与参数注释，默认 `gap_multiplier=1.5`。
        - 会话判定规则优化：若一组消息的“组内最大间隔”同时小于其首条消息的前间隔和末条消息的后间隔（两侧留白更大），该组判为一轮完整对话；相邻满足条件的段自动合并。
        - 适配快速/正常/长对话：算法对跨度更大的会话不再遗漏，统计更稳健。

        **1.3.2**
        - 在“对话模式分析”页的 IQR 分析前新增“算法与指标说明”窗口：包含原理、动态阈值公式与回退机制、切分与段合并规则、指标释义与使用建议，便于理解与复现。

        **1.3.1**
        - 取消“对话模式分析”时间轴上的半透明矩形叠加，减少渲染开销与视觉干扰；保留散点时间轴与会话统计。
        - 将“回复间隔分布”由直方图改为一维散点：横轴为间隔时长（秒），纵轴隐藏，更直观呈现整体间隔分布与密度。
        - 性能说明：减少形状绘制与悬停负载，交互缩放更流畅；本次改动作为 1.3.1 Patch 生效。

        **1.3.0**
        - 新增“会话轮数”切分算法为 IQR 自适应间隔（Adaptive IQR）：
          - 所有数据按 `Create_time`（时间戳）升序排列；计算相邻消息间隔 `delta_t`（秒）。
          - 动态阈值：`Q1,Q3`=25%/75%分位数；`IQR = Q3 - Q1`；`gap_threshold = Q3 + gap_multiplier × IQR`。
          - 回退：当 `IQR=0` 或数据极度稀疏（样本少、极端集中），阈值改用“平均间隔的 2 倍”。
          - 初步切分：当相邻两条消息间隔 `> gap_threshold` 时，判定开启新一轮对话。
          - 段内类型：若段内存在两个不同的 `is_sender` → 标记为“**双人互动**”；否则 → “**单人独白**”。
          - 短间隔段合并：若相邻两段间隔 `< gap_threshold/2` 且至少一段为“双人互动”，则合并，并重算起止与类型。
        - 结果输出与可视化：
          - 统计轮数（`len(conversations)`）、每段起止时间、持续时长（分钟）、段类型；
          - 时间轴散点图：不同 `is_sender` 用不同颜色；
          - 会话段以半透明色块显示：绿色=“双人互动”，蓝色=“单人独白”。
        - UI 改动：
          - “对话模式分析”页参数切换为 `IQR倍数（gap_multiplier）`，默认 `1.5`；
          - 指标面板新增 `Q1/Q3/IQR/阈值` 展示，并说明阈值计算与回退条件。

        **1.2.3**
        - 统一“对话模式分析”和“表情包分析”会话切分算法为“滑动窗口”。
          - 移除表情包页的密度簇参数面板，改为“窗口宽度”“平均间隔阈值”。
        - 在“回复间隔分类（散点时间轴）”下新增短/中/长间隔阈值范围展示：
          - 短间隔：0～θ₁ 秒；中间隔：θ₁～θ₂ 秒；长间隔：≥θ₂ 秒；
          - 其中 θ₁、θ₂ 为相邻聚类中心的中点（单位：秒）。
        - 方法说明（已在页面下方标注）：
          - 对回复间隔做 `log1p(gap)` 变换，KMeans(k=3)聚类得到中心；
          - 按中心值升序，用相邻中心中点作为阈值；
          - 当聚类不稳定或样本不足时，回退为“间隔跳跃分组”算法。
        - 修复：表情包页控件与算法不一致（密度簇→滑动窗口），避免参数无效。

        **1.2.2**
        - 修复：“回复间隔分类”被替换为频数图的问题；恢复散点时间轴。
        - 恢复“回复间隔分类”的散点图时间轴展示：
          - x=消息时间，y=回复间隔（秒），颜色区分“短/中/长”；
          - 悬停显示回复方、类型与文本摘要；支持交互缩放。
        - 重新引入短/中/长间隔识别：
          - 先用 `log1p(gap)` 的 KMeans(k=3) 自动分类；
          - 若样本或聚类不稳定，则回退到“跳跃边界分组”算法；
          - 在图下方显示分类中心值（秒）用于参考。
        - 保留“回复间隔分布（直方图）”用于整体分布查看。

        **1.2.1**
        - 删除“混合算法”，改用“滑动窗口算法”。
          - 定义滑动窗口（如 15 分钟内的消息视为候选组）；
          - 计算窗口内消息数与连续时间间隔的平均值；
          - 若窗口内平均间隔 < 阈值且包含不同 id（sender），判为一轮对话；
          - 向前滑动并扩展，直到条件不再满足，则结束当前轮；随后从下一条消息继续判定。
        - UI 改动：
          - “对话模式分析”页参数面板切换为“窗口宽度（分钟）”“平均间隔阈值（分钟）”；
          - 仍支持以半透明色块标注会话段，并保留中心/边界参考线。

        **1.2.0**
        - 新增会话切分“混合算法”：gap阈值 + 局部密度下降 + 一维簇优化（eps=gap, min_samples=2），并保留“散点对话”（仅两条消息且间隔<gap）归为一轮。
        - 支持参数：`gap(分钟)`、`滑动窗口宽度(分钟)`、`密度下降比例`，可在“对话模式”页配置。
        - UI 更新：时间轴散点图用半透明色块标示每轮会话起止；保留中心/边界虚线参考；统计新增“双人互动会话数”“单人独白会话数”。

        **1.1.3**
        - 移除“对话模式的时间簇分析”及相关参数与可视元素：
          - 删除自动间隔聚类与簇中心参考线；回复间隔改为基础直方图展示；
          - 会话切分改为固定最大回复间隔阈值（分钟）的手动模式；
          - 修复“表情包分析”页对旧簇参数的依赖，保持统计功能正常。（Patch/UI）

        **1.1.2**
        - 统一统计口径为 `talker`：
          - 响应时间样本与中位数计算仅按 `talker` 分组，移除 `sender` 回退；
          - 平均消息长度计算为“文本总长度/文本样本数”，仅统计 `type_name=文本`，按 `talker` 的 A/B ID 分组；
          - 显示参与者名称的映射统一使用 `talker`，若未映射则直接显示其 ID 字符串。（Patch）

        **1.1.1**
        - 调整“平均消息长度”指标的计算口径（双方对比页）：
          - 仅统计 `type_name=文本` 的消息样本；
          - 按行级 `talker/ID`（派生为 `derived_sender_id`）进行分组计算；
          - 显示的“样本数”为文本消息条数；无样本时显示“N/A”。（Patch）

        **1.1.0**
        - 新增会话模式：“粗分时间+细分密度”。
          - 先按较大时间间隔（≥阈值，如30分钟）切分为粗段；
          - 每段内以小半径DBSCAN聚类；
          - 若某子段消息数<2且与相邻段间隔短（<短阈值），强制合并为一轮。

        **1.0.6**
        - 移除顶部两条说明：
          - “按行级ID构建显示参与者列：”
          - “当双方输入的显示名称相同或为空时，会导致分组合并，从而使：”
        - 清理页面顶部说明，避免冗余文本干扰视图。（Patch）

        **1.0.5**
        - 新增“更新记录”页面模块，集中展示版本化更新说明。（Patch）

        **1.0.4**
        - 移除会话切分模式中的“自动(间隔聚类)”与“混合(间隔+密度)”。（Patch/UI调整）
        - 仅保留“两阶段(DBSCAN+规则)”“密度簇中心”“DBSCAN(时间簇)”三种模式，参数面板随选择自适应。

        **1.0.3**
        - 修正小时均值统计逻辑：仅对“当天存在消息”的日期构建完整 `date × 24小时` 网格，缺失小时补零后再按小时求均值。（Patch）
        - 完全无消息的日期不参与统计，符合“按活跃日补零”的口径。

        **1.0.2**
        - 极坐标图移除自定义缩放按钮与会话内缩放比例，使用固定径向范围，保留原有渲染与交互缩放能力。（Patch）

        **1.0.1**
        - 确认“时间轴散点图”覆盖所有样本（在所选时间范围内且时间戳有效）。
        - 说明其不受“最大响应间隔”筛选影响；提出可视改进建议（显示点数、重叠优化）。
        """
    )
