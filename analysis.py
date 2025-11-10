import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from snownlp import SnowNLP
except Exception:
    SnowNLP = None

try:
    import jieba
except Exception:
    jieba = None

try:
    import emoji
except Exception:
    emoji = None

# Optional: tslearn time series clustering（用于间隔聚类分段）
try:
    from tslearn.clustering import TimeSeriesKMeans
except Exception:
    TimeSeriesKMeans = None


LINK_PAT = re.compile(r"(https?://|www\.)", re.IGNORECASE)


def normalize_type(type_name: Any, msg: Any, src: Any) -> str:
    """Normalize raw type_name from real WeChat data to a standard set.
    Returns one of:
    "文本", "图片", "动画表情", "视频", "语音", "文件", "转账", "位置",
    "系统通知", "合并转发的聊天记录", "引用回复", "(分享)小程序", "(分享)笔记", "(分享)卡片式链接", "链接", "其他".
    """
    tn = str(type_name or "").strip().lower()

    # Direct mappings by keywords
    if "引用" in tn:
        return "引用回复"
    if "文本" in tn or "text" in tn:
        return "文本"
    if ("动画表情" in tn) or ("表情" in tn) or ("emoji" in tn) or ("动图" in tn) or ("gif" in tn):
        return "动画表情"
    if ("图片" in tn) or ("image" in tn) or ("photo" in tn) or ("img" in tn):
        return "图片"
    if ("视频" in tn) or ("video" in tn):
        return "视频"
    if ("语音" in tn) or ("voice" in tn) or ("audio" in tn):
        return "语音"
    if ("文件" in tn) or ("file" in tn):
        return "文件"
    if ("转账" in tn) or ("红包" in tn) or ("收款" in tn) or ("支付" in tn) or ("transfer" in tn):
        return "转账"
    if ("位置" in tn) or ("地理" in tn) or ("定位" in tn) or ("location" in tn):
        return "位置"
    if ("系统通知" in tn) or ("系统消息" in tn) or ("撤回" in tn) or ("系统" in tn):
        return "系统通知"
    if ("合并转发" in tn) or ("聊天记录" in tn):
        return "合并转发的聊天记录"
    if ("小程序" in tn):
        return "(分享)小程序"
    if ("笔记" in tn):
        return "(分享)笔记"
    if ("卡片" in tn and "链接" in tn):
        return "(分享)卡片式链接"
    if ("链接" in tn) or ("link" in tn):
        return "链接"

    # Fallback: detect URL presence in msg/src -> 链接
    try:
        msg_text = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False)
    except Exception:
        msg_text = str(msg)
    src_text = None
    if src is not None:
        if isinstance(src, str):
            src_text = src
        else:
            try:
                src_text = json.dumps(src, ensure_ascii=False)
            except Exception:
                src_text = str(src)
    if (msg_text and LINK_PAT.search(msg_text)) or (src_text and LINK_PAT.search(src_text)):
        return "链接"

    return "其他"


def load_json_records(fp_or_str) -> List[Dict[str, Any]]:
    """加载 JSON 记录，支持：
    - 纯 JSON 数组或对象
    - JSON Lines（每行一个对象）
    - 多种常见编码：utf-8/utf-8-sig/gb18030/utf-16(le/be)/cp936/latin-1
    - 典型对象包装：messages/records/data/items/list/rows 等键包含的列表
    输入可为文件流、字节、路径字符串或已解码文本。
    """
    def _try_parse_text(txt: str) -> Optional[List[Dict[str, Any]]]:
        s = (txt or "").strip()
        if not s:
            return []
        # 1) 尝试整体 JSON
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                # 常见包装键，取其中的列表
                for k in [
                    "messages","msgs","records","data","items","list","rows","chats","conversation"
                ]:
                    if k in val and isinstance(val[k], list):
                        return val[k]
                return [val]
        except json.JSONDecodeError:
            pass
        # 2) JSON Lines
        records: List[Dict[str, Any]] = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, list):
                    # 某些 JSONL 可能包含列表行
                    for it in obj:
                        if isinstance(it, dict):
                            records.append(it)
            except json.JSONDecodeError:
                continue
        return records

    def _decode_bytes(b: bytes) -> Optional[List[Dict[str, Any]]]:
        # 依次尝试多种常见编码；严格解码，避免静默丢字节导致 JSON 破损
        encodings = [
            "utf-8", "utf-8-sig", "gb18030", "utf-16", "utf-16le", "utf-16be", "cp936", "latin-1"
        ]
        for enc in encodings:
            try:
                txt = b.decode(enc)
                parsed = _try_parse_text(txt)
                if parsed is not None and len(parsed) > 0:
                    return parsed
                # 若解析为空，但文本存在，仍返回空列表以继续尝试其他编码
            except Exception:
                continue
        # 最后回退：utf-8 严格失败后，使用 ignore 以尽量读取
        try:
            txt = b.decode("utf-8", errors="ignore")
            return _try_parse_text(txt) or []
        except Exception:
            return []

    try:
        # 文件流 / UploadedFile
        if hasattr(fp_or_str, "read"):
            try:
                # 复位光标，避免此前读取导致空数据
                if hasattr(fp_or_str, "seek"):
                    fp_or_str.seek(0)
            except Exception:
                pass
            data = fp_or_str.read()
            if isinstance(data, bytes):
                return _decode_bytes(data) or []
            else:
                # 已是字符串
                return _try_parse_text(str(data)) or []
        # 字节或字符串（可能是路径或已解码文本）
        elif isinstance(fp_or_str, (bytes, str)):
            # 字节：按编码表解析
            if isinstance(fp_or_str, bytes):
                return _decode_bytes(fp_or_str) or []
            s = str(fp_or_str)
            # 先尝试按路径读取二进制
            p = Path(s)
            if p.exists() and p.is_file():
                try:
                    with open(p, "rb") as f:
                        b = f.read()
                    return _decode_bytes(b) or []
                except Exception:
                    # 回退：按文本解析
                    return _try_parse_text(s) or []
            else:
                # 非路径：直接按文本解析
                return _try_parse_text(s) or []
        else:
            raise ValueError("Unsupported input type for JSON loading")
    except Exception:
        # 任意异常统一回退为空列表，避免中断应用
        return []
    


def is_link_message(type_name: str, msg: Optional[str], src: Optional[str]) -> bool:
    # Use normalized type to include shares as link-like content
    tnorm = normalize_type(type_name, msg, src)
    if tnorm in {"链接", "(分享)小程序", "(分享)笔记", "(分享)卡片式链接"}:
        return True
    # Fallback: URL pattern
    if msg and LINK_PAT.search(str(msg)):
        return True
    if src is not None:
        if isinstance(src, str):
            if LINK_PAT.search(src):
                return True
        elif isinstance(src, (dict, list)):
            try:
                src_text = json.dumps(src, ensure_ascii=False)
                if LINK_PAT.search(src_text):
                    return True
            except Exception:
                pass
    return False


def extract_emojis(text: str) -> List[str]:
    if not text:
        return []
    if emoji is None:
        # Fallback: naive unicode range for emoji
        # This is rough; recommend installing 'emoji'
        return [ch for ch in text if ord(ch) > 0x1F000]
    return [ch for ch in text if ch in emoji.EMOJI_DATA]


def week_of_month(dt: pd.Timestamp) -> int:
    first_day = dt.replace(day=1)
    dom = dt.day
    return int(math.ceil((dom + first_day.weekday()) / 7.0))


def prepare_dataframe(records: List[Dict[str, Any]], me_sender_flag: int = 1) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Normalize columns
    for col in ["id", "MsgSvrID", "type_name", "is_sender", "talker", "room_name", "msg", "src", "extra", "CreateTime"]:
        if col not in df.columns:
            df[col] = None

    # Parse datetime (robust): support string timestamps and epoch seconds/milliseconds
    def _parse_ct(val):
        try:
            if pd.isna(val):
                return pd.NaT
            # numeric epoch
            if isinstance(val, (int, float)):
                n = int(val)
                # heuristics: us vs ms vs s
                if n >= 10**14:  # microseconds
                    return pd.to_datetime(n, unit="us", errors="coerce")
                elif n >= 10**12:  # milliseconds
                    return pd.to_datetime(n, unit="ms", errors="coerce")
                elif n >= 10**9:   # seconds
                    return pd.to_datetime(n, unit="s", errors="coerce")
                else:
                    # fallback general
                    return pd.to_datetime(n, errors="coerce")
            # string
            s = str(val).strip()
            if not s:
                return pd.NaT
            if s.isdigit():
                n = int(s)
                if len(s) >= 16:  # microseconds
                    return pd.to_datetime(n, unit="us", errors="coerce")
                elif len(s) >= 13:  # milliseconds
                    return pd.to_datetime(n, unit="ms", errors="coerce")
                else:  # seconds
                    return pd.to_datetime(n, unit="s", errors="coerce")
            # explicit common formats
            fmt_candidates = [
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M",
            ]
            for fmt in fmt_candidates:
                try:
                    dt = datetime.strptime(s, fmt)
                    return pd.Timestamp(dt)
                except Exception:
                    pass
            # general parsing
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT

    df["CreateTime"] = df["CreateTime"].apply(_parse_ct)
    df = df.dropna(subset=["CreateTime"]).sort_values("CreateTime").reset_index(drop=True)

    # Sender labels
    df["is_sender"] = pd.to_numeric(df["is_sender"], errors="coerce").fillna(0).astype(int)
    df["sender"] = np.where(df["is_sender"] == me_sender_flag, "我", "对方")

    # Message features
    df["type_name"] = df["type_name"].fillna("").astype(str)
    df["msg"] = df["msg"].apply(lambda v: "" if pd.isna(v) else (v if isinstance(v, str) else str(v)))
    df["src"] = df["src"].apply(lambda v: None if pd.isna(v) else v)
    # Normalized type
    df["type_norm"] = df.apply(lambda r: normalize_type(r["type_name"], r["msg"], r["src"]), axis=1)
    df["message_length"] = df["msg"].apply(lambda x: len(str(x)))
    df["is_link"] = df.apply(lambda r: is_link_message(r["type_name"], r["msg"], r["src"]), axis=1)

    # Time breakdown
    df["date"] = df["CreateTime"].dt.date
    df["hour"] = df["CreateTime"].dt.hour
    df["weekday"] = df["CreateTime"].dt.weekday  # Monday=0
    df["month"] = df["CreateTime"].dt.to_period("M").astype(str)
    df["week_in_month"] = df["CreateTime"].apply(week_of_month)
    df["month_week_label"] = df["month"] + " W" + df["week_in_month"].astype(str)

    # Emojis
    df["emojis"] = df["msg"].apply(extract_emojis)

    return df


def filter_by_date(df: pd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        df = df[df["CreateTime"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["CreateTime"] <= pd.Timestamp(end)]
    return df


def overview_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    sender_counts = df["sender"].value_counts().to_dict()
    type_counts = df["type_norm"].value_counts().to_dict()
    dow_counts = df["weekday"].value_counts().sort_index().to_dict()
    # Heatmap pivot: day-of-week vs month-week label
    pivot = df.groupby(["month_week_label", "weekday"]).size().unstack(fill_value=0)
    # Time-of-day avg per day over 24 hourly slots (0-1h, 1-2h, ...)
    # Requirement: Exclude days with absolutely no messages; but for days that have
    # messages (in any hour), include zero counts for hours without messages when averaging.
    # 1) Identify dates that have any messages in the filtered dataframe
    active_dates = df["date"].dropna().unique()
    # 2) Build full grid of (date, hour) for those active dates
    hours = np.arange(24)
    grid = pd.MultiIndex.from_product([active_dates, hours], names=["date", "hour"]).to_frame(index=False)
    # 3) Aggregate actual counts per (date, hour)
    per_day_hour = df.groupby(["date", "hour"]).size().reset_index(name="count")
    # 4) Left join counts onto the full grid and fill missing as 0
    full_counts = grid.merge(per_day_hour, on=["date", "hour"], how="left")
    full_counts["count"] = full_counts["count"].fillna(0)
    # 5) Average across dates per hour (zeros included for hours with no messages on active days)
    avg_slot = full_counts.groupby("hour")["count"].mean().reindex(range(24), fill_value=0)
    slot_labels = [f"{i:02d}:00-{(i+1)%24:02d}:00" for i in range(24)]
    return {
        "sender_counts": sender_counts,
        "type_counts": type_counts,
        "dow_counts": dow_counts,
        "heatmap_pivot": pivot,
        "avg_slot": avg_slot,
        "slot_labels": slot_labels,
    }


def response_time_stats(df: pd.DataFrame, max_reply_hours: int = 24) -> Dict[str, Any]:
    if df.empty:
        return {}

    max_delta = timedelta(hours=max_reply_hours)

    def compute_avg_reply(from_label: str, to_label: str) -> Tuple[float, float, int]:
        # Adjacent-only reply intervals: count gap only when sender switches from from_label to to_label
        deltas = []
        for i in range(1, len(df)):
            s_prev = str(df.iloc[i-1]["sender"]) 
            s_curr = str(df.iloc[i]["sender"]) 
            if s_prev == from_label and s_curr == to_label:
                delta = df.iloc[i]["CreateTime"] - df.iloc[i-1]["CreateTime"]
                if timedelta(0) < delta <= max_delta:
                    deltas.append(delta.total_seconds())
        if not deltas:
            return (float("nan"), float("nan"), 0)
        return (float(np.mean(deltas)), float(np.median(deltas)), len(deltas))

    me_mean, me_median, me_n = compute_avg_reply("我", "对方")
    you_mean, you_median, you_n = compute_avg_reply("对方", "我")
    return {
        "me_to_you": {"mean_sec": me_mean, "median_sec": me_median, "n": me_n},
        "you_to_me": {"mean_sec": you_mean, "median_sec": you_median, "n": you_n},
    }


def response_time_samples(df: pd.DataFrame, max_reply_hours: int = 24) -> Dict[str, Any]:
    """Return raw reply time samples (seconds) for both directions.
    me_to_you: time from my message to next message by the other side
    you_to_me: time from their message to my next message
    """
    if df.empty:
        return {"me_to_you": [], "you_to_me": []}

    max_delta = timedelta(hours=max_reply_hours)

    def compute_deltas_adjacent(from_label: str, to_label: str) -> List[float]:
        deltas: List[float] = []
        for i in range(1, len(df)):
            s_prev = str(df.iloc[i-1]["sender"]) 
            s_curr = str(df.iloc[i]["sender"]) 
            if s_prev == from_label and s_curr == to_label:
                delta = df.iloc[i]["CreateTime"] - df.iloc[i-1]["CreateTime"]
                if timedelta(0) < delta <= max_delta:
                    deltas.append(delta.total_seconds())
        return deltas

    return {
        "me_to_you": compute_deltas_adjacent("我", "对方"),
        "you_to_me": compute_deltas_adjacent("对方", "我"),
    }


def response_time_samples_by_labels(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
    sender_col: str = "sender",
    max_reply_hours: int = 24,
) -> Dict[str, Any]:
    """Generic reply time samples using a custom sender column and two labels.
    Returns a_to_b (from label_a to the next message by label_b) and b_to_a.
    """
    if df.empty or sender_col not in df.columns:
        return {"a_to_b": [], "b_to_a": []}

    max_delta = timedelta(hours=max_reply_hours)
    col = sender_col

    def compute_deltas_adjacent(from_label: str, to_label: str) -> List[float]:
        deltas: List[float] = []
        for i in range(1, len(df)):
            s_prev = str(df.iloc[i-1][col])
            s_curr = str(df.iloc[i][col])
            if s_prev == from_label and s_curr == to_label:
                delta = df.iloc[i]["CreateTime"] - df.iloc[i-1]["CreateTime"]
                if timedelta(0) < delta <= max_delta:
                    deltas.append(delta.total_seconds())
        return deltas

    return {
        "a_to_b": compute_deltas_adjacent(label_a, label_b),
        "b_to_a": compute_deltas_adjacent(label_b, label_a),
    }


def sessionize(df: pd.DataFrame, idle_minutes: int = 30) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sessions = []
    current = {"messages": [], "start": None, "end": None}
    idle = timedelta(minutes=idle_minutes)
    prev_time = None
    for _, row in df.iterrows():
        t = row["CreateTime"]
        if prev_time is None or (t - prev_time) <= idle:
            if not current["messages"]:
                current["start"] = t
            current["messages"].append(row)
            current["end"] = t
        else:
            sessions.append(current)
            current = {"messages": [row], "start": t, "end": t}
        prev_time = t
    if current["messages"]:
        sessions.append(current)
    return sessions


def _kmeans_1d(values: np.ndarray, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D K-Means for k=2 on log-transformed gaps.
    Returns centers and labels.
    """
    if values.size == 0:
        return np.array([]), np.array([])
    # k=2 centers init: min & max
    c0, c1 = float(np.min(values)), float(np.max(values))
    if abs(c1 - c0) < 1e-9:
        # all equal
        centers = np.array([c0, c1])
        labels = np.zeros_like(values, dtype=int)
        return centers, labels
    centers = np.array([c0, c1], dtype=float)
    labels = np.zeros_like(values, dtype=int)
    for _ in range(max_iter):
        # assign
        d0 = np.abs(values - centers[0])
        d1 = np.abs(values - centers[1])
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update centers
        for k in (0, 1):
            mask = labels == k
            if np.any(mask):
                centers[k] = float(np.mean(values[mask]))
    # ensure centers order: ascending
    order = np.argsort(centers)
    centers = centers[order]
    # remap labels accordingly
    remap = {int(order[0]): 0, int(order[1]): 1}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    return centers, labels


def _kmeans_1d_k(values: np.ndarray, k: int = 2, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """General 1D K-Means on log-transformed values for arbitrary k.
    Returns centers and labels (centers ascending, labels remapped accordingly).
    """
    if values.size == 0 or k <= 1:
        return np.array([]), np.array([])
    # init centers by quantiles for stability
    qs = [(i + 0.5) / float(k) for i in range(k)]
    centers = np.array([float(np.quantile(values, q)) for q in qs], dtype=float)
    labels = np.zeros_like(values, dtype=int)
    for _ in range(max_iter):
        # assign
        dists = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(dists, axis=1).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = float(np.mean(values[mask]))
    # sort centers ascending and remap labels
    order = np.argsort(centers)
    centers = centers[order]
    remap = {int(order[i]): i for i in range(k)}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    return centers, labels


def sessionize_by_clustering(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cluster time gaps (CreateTime deltas) into two groups to auto split sessions.
    Uses log1p(gap_seconds) and k-means (k=2).
    Returns (sessions, cluster_info).
    """
    if df.empty:
        return [], {"status": "empty"}
    times = df["CreateTime"].tolist()
    if len(times) <= 1:
        s = {"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[0]}
        return [s], {"status": "single", "session_count": 1}

    # compute gaps in seconds between consecutive messages
    gaps_sec = []
    for i in range(1, len(times)):
        delta = (times[i] - times[i - 1]).total_seconds()
        gaps_sec.append(max(0.0, float(delta)))

    values = np.log1p(np.array(gaps_sec, dtype=float))
    if np.allclose(values, values[0]):
        # all gaps same -> one session
        s = {"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[-1]}
        info = {
            "status": "degenerate",
            "centers_sec": [float(np.expm1(values[0]))] * 2,
            "threshold_sec": float(np.expm1(values[0])),
            "break_ratio": 0.0,
            "session_count": 1,
        }
        return [s], info

    centers, labels = _kmeans_1d(values)
    if centers.size < 2:
        s = {"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[-1]}
        return [s], {"status": "fallback", "session_count": 1}

    # centers[0] = small-gaps, centers[1] = large-gaps
    thr_v = float((centers[0] + centers[1]) / 2.0)
    thr_sec = float(np.expm1(thr_v))
    centers_sec = [float(np.expm1(c)) for c in centers.tolist()]

    # build sessions using labels: label 1 indicates break between i-1 and i
    sessions = []
    start_idx = 0
    breaks = 0
    for i in range(1, len(times)):
        if labels[i - 1] == 1:
            # end session at i-1
            part = df.iloc[start_idx:i]
            sessions.append({"messages": [row for _, row in part.iterrows()], "start": times[start_idx], "end": times[i - 1]})
            start_idx = i
            breaks += 1
    # tail
    part = df.iloc[start_idx:len(times)]
    sessions.append({"messages": [row for _, row in part.iterrows()], "start": times[start_idx], "end": times[-1]})

    info = {
        "status": "ok",
        "centers_sec": centers_sec,
        "threshold_sec": thr_sec,
        "break_ratio": float(breaks / max(1, len(gaps_sec))),
        "session_count": len(sessions),
    }
    return sessions, info


def compute_reply_gaps(df: pd.DataFrame) -> List[float]:
    """Return list of consecutive reply gaps in seconds."""
    if df.empty:
        return []
    times = df["CreateTime"].tolist()
    gaps_sec = []
    for i in range(1, len(times)):
        delta = (times[i] - times[i - 1]).total_seconds()
        gaps_sec.append(max(0.0, float(delta)))
    return gaps_sec


def cluster_gaps(gaps_sec: List[float]) -> Dict[str, Any]:
    """Cluster reply gaps via 1D k-means (k=2) on log1p(gap)."""
    if not gaps_sec:
        return {"status": "empty", "labels": [], "centers_sec": []}
    values = np.log1p(np.array(gaps_sec, dtype=float))
    centers, labels = _kmeans_1d(values)
    centers_sec = [float(np.expm1(c)) for c in centers.tolist()] if centers.size else []
    return {
        "status": "ok" if centers.size else "fallback",
        "labels": labels.tolist() if centers.size else [0] * len(gaps_sec),
        "centers_sec": centers_sec,
    }


def cluster_gaps_multilevel(gaps_sec: List[float], k: int = 2) -> Dict[str, Any]:
    """Cluster reply gaps into k levels using k-means on log1p(gap)."""
    if not gaps_sec:
        return {"status": "empty", "labels": [], "centers_sec": []}
    values = np.log1p(np.array(gaps_sec, dtype=float))
    centers, labels = _kmeans_1d_k(values, k=max(2, int(k)))
    centers_sec = [float(np.expm1(c)) for c in centers.tolist()] if centers.size else []
    return {
        "status": "ok" if centers.size else "fallback",
        "labels": labels.tolist() if centers.size else [0] * len(gaps_sec),
        "centers_sec": centers_sec,
    }


def cluster_gaps_by_jumps(
    gaps_sec: List[float],
    min_group_size: int = 5,
    ratio_threshold: float = 3.0,
    diff_quantile: float = 0.9,
    target_groups: Optional[int] = None,
) -> Dict[str, Any]:
    """Group sorted reply gaps by local jump boundaries.
    - Sort gaps ascending
    - Compute consecutive diffs
    - Mark boundary at position i+1 when right_diff / max(eps, left_diff) >= ratio_threshold
      and right_diff >= quantile(diff, diff_quantile)
    - Enforce min_group_size by pruning boundaries too close
    Returns labels (original order), labels_sorted, group_centers_sec, bound_positions_sorted, bound_values_sec
    """
    n = len(gaps_sec)
    if n == 0:
        return {"status": "empty", "labels": [], "labels_sorted": [], "group_centers_sec": [], "bound_positions_sorted": [], "bound_values_sec": []}
    if n == 1:
        return {"status": "single", "labels": [0], "labels_sorted": [0], "group_centers_sec": [float(gaps_sec[0])], "bound_positions_sorted": [], "bound_values_sec": []}

    sort_idx = np.argsort(gaps_sec)
    gaps_sorted = np.array(gaps_sec, dtype=float)[sort_idx]
    # consecutive diffs
    diffs = np.diff(gaps_sorted)  # length n-1, diff[i] = gaps_sorted[i+1] - gaps_sorted[i]
    eps = 1e-9
    # quantile threshold for right diffs
    try:
        diff_thr = float(np.quantile(diffs, diff_quantile)) if diffs.size > 0 else 0.0
    except Exception:
        diff_thr = 0.0

    # candidate boundaries positions (on sorted index space), boundary at i+1
    candidates = []
    scores = []  # jump scores for each potential boundary (i+1)
    for i in range(1, n - 1):
        left_diff = max(eps, gaps_sorted[i] - gaps_sorted[i - 1])
        right_diff = gaps_sorted[i + 1] - gaps_sorted[i]
        if right_diff <= 0:
            continue
        ratio = right_diff / left_diff
        score = ratio  # primary score; could be combined with right_diff
        if (ratio >= ratio_threshold) and (right_diff >= diff_thr):
            candidates.append(i + 1)
        scores.append((i + 1, score))

    # boundary selection
    bounds = []
    if target_groups is not None and isinstance(target_groups, int) and target_groups >= 2:
        # Greedy pick top (target_groups-1) by score with spacing constraint
        need = target_groups - 1
        # sort by score desc
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sel = []
        for pos, sc in sorted_scores:
            # enforce spacing
            if not sel or min(abs(pos - s) for s in sel) >= min_group_size:
                sel.append(pos)
                if len(sel) >= need:
                    break
        bounds = sorted(sel)
        # ensure tail group has min size; if not, drop last
        if bounds and (n - bounds[-1]) < min_group_size:
            bounds.pop()
    else:
        # prune candidates by spacing
        last = 0
        for b in sorted(candidates):
            if (b - last) >= min_group_size:
                bounds.append(b)
                last = b
        # ensure tail group also has min size; if not, drop last boundary
        if bounds:
            if (n - bounds[-1]) < min_group_size:
                bounds.pop()

    # build labels on sorted order
    labels_sorted = np.zeros(n, dtype=int)
    current = 0
    prev = 0
    for b in bounds:
        labels_sorted[prev:b] = current
        prev = b
        current += 1
    labels_sorted[prev:n] = current

    # centers per preliminary group
    centers = []
    start = 0
    for b in bounds + [n]:
        group_vals = gaps_sorted[start:b]
        if group_vals.size > 0:
            centers.append(float(np.median(group_vals)))
        start = b

    # refinement: assign by center mid-thresholds to avoid misassignment at tails
    if centers:
        centers_arr = np.array(centers, dtype=float)
        # thresholds between adjacent centers
        if len(centers_arr) > 1:
            mids_thr = (centers_arr[:-1] + centers_arr[1:]) / 2.0  # len = k-1
            # contiguous assignment by thresholds (on sorted gaps)
            new_labels_sorted = np.searchsorted(mids_thr, gaps_sorted)
            # enforce min_group_size by merging tiny groups into nearest neighbor
            # compute group sizes
            k = int(centers_arr.size)
            sizes = [int(np.sum(new_labels_sorted == i)) for i in range(k)]
            changed = True
            while changed:
                changed = False
                for i in range(k):
                    if sizes[i] > 0 and sizes[i] < min_group_size:
                        # merge into closer neighbor by center distance
                        if i == 0 and k > 1:
                            target = 1
                        elif i == k - 1 and k > 1:
                            target = k - 2
                        else:
                            dl = abs(centers_arr[i] - centers_arr[i - 1])
                            dr = abs(centers_arr[i] - centers_arr[i + 1])
                            target = i - 1 if dl <= dr else i + 1
                        new_labels_sorted[new_labels_sorted == i] = target
                        sizes[target] += sizes[i]
                        sizes[i] = 0
                        changed = True
                # recompute centers after merges
                centers_arr = np.array([
                    float(np.median(gaps_sorted[new_labels_sorted == i])) if np.any(new_labels_sorted == i) else float('nan')
                    for i in range(k)
                ], dtype=float)
            # compact labels to contiguous 0..m-1 in sorted order
            unique_labels = sorted(set(int(x) for x in new_labels_sorted.tolist()))
            remap = {old: idx for idx, old in enumerate(unique_labels)}
            new_labels_sorted = np.array([remap[int(x)] for x in new_labels_sorted.tolist()], dtype=int)
            # recompute centers per final groups
            final_centers = []
            for i in range(len(unique_labels)):
                vals = gaps_sorted[new_labels_sorted == i]
                if vals.size > 0:
                    final_centers.append(float(np.median(vals)))
            centers = final_centers
            labels_sorted = new_labels_sorted

    # map labels back to original order
    labels_orig = np.zeros(n, dtype=int)
    labels_orig[sort_idx] = labels_sorted

    # boundaries where label changes (sorted space)
    bounds_refined = []
    prev_label = int(labels_sorted[0])
    for i in range(1, n):
        if int(labels_sorted[i]) != prev_label:
            bounds_refined.append(i)
            prev_label = int(labels_sorted[i])
    bound_values = [float(gaps_sorted[pos]) for pos in bounds_refined]

    return {
        "status": "ok",
        "labels": labels_orig.tolist(),
        "labels_sorted": labels_sorted.tolist(),
        "group_centers_sec": centers,
        "bound_positions_sorted": bounds_refined,
        "bound_values_sec": bound_values,
        "diff_threshold": diff_thr,
    }


def sessionize_hybrid(
    df: pd.DataFrame,
    max_gap_minutes: int = 10,
    window_minutes: int = 10,
    density_quantile: float = 0.25,
    density_drop_ratio: float = 0.3,
    local_window: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    混合算法（Hybrid）：
    - 基本阈值：若相邻两条消息间隔 > gap(分钟)，判定为新会话边界；
    - 局部密度辅助：对时间戳使用滑动窗口密度估计，若连续窗口密度显著低于局部平均（如 < drop_ratio × 局部均值），
      且对应的相邻间隔未超过 gap，则在该连续段的最低密度点处切分；
    - 簇优化：在时间轴上按 eps=gap、min_samples=2 的一维聚类，若边界将同簇消息分裂则移除该边界；
      对仅两条且间隔 < gap 的散点对话也归入一轮；
    - 段类型识别：标注“双人互动”与“单人独白”。
    """
    if df.empty:
        return [], {"status": "empty"}
    times = df["CreateTime"].tolist()
    n = len(times)
    if n == 1:
        return [{"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[0]}], {
            "status": "single", "session_count": 1
        }
    # compute gaps in seconds
    gaps_sec = compute_reply_gaps(df)

    # midpoints (seconds) for each gap i between i-1 and i
    times_sec = [float(pd.Timestamp(t).timestamp()) for t in times]
    mids_sec = [ (times_sec[i-1] + times_sec[i]) / 2.0 for i in range(1, n) ]

    # sliding-window density using two-pointer for efficiency
    gap_seconds = float(max_gap_minutes) * 60.0
    half_window_sec = float(window_minutes) * 60.0 / 2.0
    densities: List[float] = []  # messages per minute
    l = 0
    r = -1
    for i in range(len(mids_sec)):
        left = mids_sec[i] - half_window_sec
        right = mids_sec[i] + half_window_sec
        # advance left pointer
        while l < n and times_sec[l] < left:
            l += 1
        # advance right pointer
        if r < l - 1:
            r = l - 1
        while r + 1 < n and times_sec[r + 1] <= right:
            r += 1
        cnt = max(0, r - l + 1)
        densities.append(cnt / max(1.0, float(window_minutes)))

    # local moving average of densities
    dens_ser = pd.Series(densities, dtype=float)
    local_mean_ser = dens_ser.rolling(window=max(1, int(local_window)), center=True, min_periods=1).mean()
    local_mean = local_mean_ser.to_numpy()

    # initial boundaries by hard gap rule
    boundaries: List[int] = []  # index i where boundary between i-1 and i
    for i, g in enumerate(gaps_sec, start=1):
        if g > gap_seconds:
            boundaries.append(i)

    # density-drop assisted boundaries (within segments not exceeding gap)
    mask_low = [False] * len(densities)
    for i in range(len(densities)):
        lm = float(local_mean[i]) if i < len(local_mean) else float('nan')
        if lm > 0 and (densities[i] < density_drop_ratio * lm) and (gaps_sec[i] <= gap_seconds):
            mask_low[i] = True
    # select lowest-density point in each contiguous low region
    j = 0
    while j < len(mask_low):
        if not mask_low[j]:
            j += 1
            continue
        k = j
        # extend run
        while k + 1 < len(mask_low) and mask_low[k + 1]:
            k += 1
        # choose idx with minimal density in [j..k]
        run = densities[j:k+1]
        if run:
            min_rel_idx = int(np.argmin(run))
            cand = j + min_rel_idx + 1  # boundary index i corresponds to gap position +1
            # only add if not already a hard boundary
            if cand not in boundaries:
                boundaries.append(cand)
        j = k + 1

    # clustering assistance: group contiguous points with consecutive gaps <= eps
    cluster_labels = [-1] * n
    cid = 0
    start = 0
    while start < n:
        end = start
        while end + 1 < n and (times_sec[end + 1] - times_sec[end]) <= gap_seconds:
            end += 1
        group_len = end - start + 1
        if group_len >= 2:
            for idx in range(start, end + 1):
                cluster_labels[idx] = cid
            cid += 1
        start = end + 1

    # remove boundaries that split same-time cluster
    refined = []
    for b in sorted(set(boundaries)):
        if 0 < b < n:
            left_c = cluster_labels[b - 1]
            right_c = cluster_labels[b]
            if (left_c != -1) and (left_c == right_c):
                # boundary splits a cluster -> drop
                continue
        refined.append(b)
    refined = sorted(set(refined))

    # build sessions from refined boundaries
    sessions = []
    start_idx = 0
    for i in refined:
        part = df.iloc[start_idx:i]
        msgs = [row for _, row in part.iterrows()]
        center_time = pd.Series([m["CreateTime"] for m in msgs]).median() if msgs else times[start_idx]
        # classify interaction vs monologue
        senders = [str(m.get("sender")) for m in msgs]
        uniq = set(senders)
        is_mono = (len(uniq) == 1)
        duo_short = False
        for t_idx in range(1, len(msgs)):
            if str(msgs[t_idx]["sender"]) != str(msgs[t_idx - 1]["sender"]) and gaps_sec[t_idx] < gap_seconds:
                duo_short = True
                break
        sessions.append({
            "messages": msgs,
            "start": times[start_idx],
            "end": times[i - 1],
            "center": center_time,
            "type": "单人独白段" if is_mono else ("双人互动" if duo_short else "杂合段"),
        })
        start_idx = i
    # tail
    part = df.iloc[start_idx:n]
    msgs = [row for _, row in part.iterrows()]
    center_time = pd.Series([m["CreateTime"] for m in msgs]).median() if msgs else times[start_idx]
    senders = [str(m.get("sender")) for m in msgs]
    uniq = set(senders)
    is_mono = (len(uniq) == 1)
    duo_short = False
    for t_idx in range(1, len(msgs)):
        if str(msgs[t_idx]["sender"]) != str(msgs[t_idx - 1]["sender"]) and gaps_sec[t_idx] < gap_seconds:
            duo_short = True
            break
    sessions.append({
        "messages": msgs,
        "start": times[start_idx],
        "end": times[-1],
        "center": center_time,
        "type": "单人独白段" if is_mono else ("双人互动" if duo_short else "杂合段"),
    })

    info = {
        "status": "ok",
        "session_count": len(sessions),
        "bound_indices": refined,
        "density_per_min": densities,
        "local_mean_per_min": local_mean_ser.tolist(),
        "window_minutes": int(window_minutes),
        "max_gap_minutes": int(max_gap_minutes),
        "density_drop_ratio": float(density_drop_ratio),
        "cluster_centers": [s.get("center") for s in sessions],
        "session_bounds": [(s.get("start"), s.get("end")) for s in sessions],
        "session_types": [s.get("type") for s in sessions],
    }
    return sessions, info


def sessionize_density_clusters(
    df: pd.DataFrame,
    window_minutes: int = 10,
    density_quantile: float = 0.25,
    min_messages: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Segment sessions by contiguous dense regions; each dense region is one session.
    Returns sessions and info including cluster centers (median time per session).
    """
    if df.empty:
        return [], {"status": "empty"}
    times = df["CreateTime"].tolist()
    n = len(times)
    if n == 1:
        return [{"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[0], "center": times[0]}], {
            "status": "single", "session_count": 1
        }
    # mid densities
    mids = [times[i - 1] + (times[i] - times[i - 1]) / 2 for i in range(1, n)]
    half = timedelta(minutes=window_minutes / 2.0)
    densities = []
    for m in mids:
        left = m - half
        right = m + half
        cnt = int(np.sum([(left <= t <= right) for t in times]))
        densities.append(cnt / max(1.0, window_minutes))
    dens = np.array(densities, dtype=float)
    dens_thr = float(np.quantile(dens, density_quantile)) if len(dens) > 0 else 0.0

    # dense flags on gaps between messages
    dense_flags = [False] + [densities[i - 1] >= dens_thr for i in range(1, n)]
    # build sessions by contiguous dense_flags True; when False indicates potential boundary
    sessions = []
    start_idx = 0
    for i in range(1, n):
        if not dense_flags[i]:
            part = df.iloc[start_idx:i]
            if len(part) >= min_messages:
                center_time = part["CreateTime"].median()
                sessions.append({"messages": [row for _, row in part.iterrows()], "start": times[start_idx], "end": times[i - 1], "center": center_time})
            start_idx = i
    # tail
    part = df.iloc[start_idx:n]
    if len(part) >= min_messages:
        center_time = part["CreateTime"].median()
        sessions.append({"messages": [row for _, row in part.iterrows()], "start": times[start_idx], "end": times[-1], "center": center_time})

    info = {
        "status": "ok",
        "session_count": len(sessions),
        "density_threshold": dens_thr,
        "window_minutes": window_minutes,
        "cluster_centers": [s["center"] for s in sessions],
    }
    return sessions, info


def sessionize_dbscan(
    df: pd.DataFrame,
    eps_seconds: int = 300,
    min_samples: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cluster messages on the time axis using a simple 1D-DBSCAN.
    - eps_seconds: neighborhood radius in seconds
    - min_samples: minimum number of points to form a core (including itself)
    Returns sessions (each cluster as one session) and cluster info.
    """
    if df.empty:
        return [], {"status": "empty"}
    times = df["CreateTime"].tolist()
    n = len(times)
    if n == 1:
        return [{"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[0], "center": times[0]}], {
            "status": "single", "session_count": 1, "dbscan_eps_sec": eps_seconds, "dbscan_min_samples": min_samples
        }

    # convert to seconds (epoch) for numeric operations
    tsec = np.array([float(t.value) * 1e-9 for t in times], dtype=float)

    # find core points using two-pointer window
    core = np.zeros(n, dtype=bool)
    left = 0
    right = 0
    for i in range(n):
        # move left pointer to maintain window within eps
        while left < n and (tsec[i] - tsec[left]) > float(eps_seconds):
            left += 1
        # move right pointer forward as long as within eps
        while right < n - 1 and (tsec[right + 1] - tsec[i]) <= float(eps_seconds):
            right += 1
        count = (right - left + 1)
        if count >= int(min_samples):
            core[i] = True
        # shrink right if it is behind i for the next iteration
        if right < i:
            right = i

    core_indices = np.where(core)[0].tolist()
    if not core_indices:
        # no clusters
        return [], {"status": "no_core", "dbscan_eps_sec": eps_seconds, "dbscan_min_samples": min_samples, "session_count": 0}

    # group contiguous core points into clusters if successive core points are within eps
    clusters_bounds_core: List[Tuple[int, int]] = []  # (first_core_idx, last_core_idx) in index space
    start_core = core_indices[0]
    prev_core = core_indices[0]
    for idx in core_indices[1:]:
        if (tsec[idx] - tsec[prev_core]) <= float(eps_seconds):
            prev_core = idx
        else:
            clusters_bounds_core.append((start_core, prev_core))
            start_core = idx
            prev_core = idx
    clusters_bounds_core.append((start_core, prev_core))

    # expand each core cluster by eps to include border points
    sessions: List[Dict[str, Any]] = []
    centers: List[pd.Timestamp] = []
    for c_start, c_end in clusters_bounds_core:
        t_left = tsec[c_start] - float(eps_seconds)
        t_right = tsec[c_end] + float(eps_seconds)
        L = int(np.searchsorted(tsec, t_left, side="left"))
        R = int(np.searchsorted(tsec, t_right, side="right") - 1)
        L = max(0, L)
        R = min(n - 1, R)
        part = df.iloc[L:R+1]
        if len(part) > 0:
            center_time = part["CreateTime"].median()
            sessions.append({
                "messages": [row for _, row in part.iterrows()],
                "start": times[L],
                "end": times[R],
                "center": center_time,
            })
            centers.append(center_time)

    info = {
        "status": "ok",
        "session_count": len(sessions),
        "dbscan_eps_sec": int(eps_seconds),
        "dbscan_min_samples": int(min_samples),
        "cluster_centers": centers,
    }
    return sessions, info


def sessionize_dbscan_rule(
    df: pd.DataFrame,
    eps_seconds: int = 300,
    min_samples: int = 3,
    consecutive_same_sender_limit: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Two-stage sessionization:
    1) DBSCAN on time axis to find major dense clusters;
    2) For messages not covered by DBSCAN clusters, apply local rule-based micro-clustering
       using a gap threshold derived from jump grouping with target_groups=3.
    Rules:
      - New round start when sender changes compared to previous message or gap exceeds threshold
      - Consecutive same-sender messages extend current round
      - End current round when gap exceeds threshold or consecutive same-sender count > limit
    """
    if df.empty:
        return [], {"status": "empty"}

    # Step 1: DBSCAN to get major clusters
    major_sessions, major_info = sessionize_dbscan(df, eps_seconds=eps_seconds, min_samples=min_samples)
    covered_idx: set = set()
    for s in major_sessions:
        for row in s.get("messages", []):
            try:
                idx = int(row.name) if hasattr(row, "name") else None
                if idx is not None:
                    covered_idx.add(idx)
            except Exception:
                continue

    # Step 2: derive gap threshold from jump grouping (target_groups=3)
    from typing import Optional
    gaps_sec = compute_reply_gaps(df)
    cl = cluster_gaps_by_jumps(gaps_sec, min_group_size=5, ratio_threshold=3.0, diff_quantile=0.90, target_groups=3)
    labels = cl.get("labels", [])
    gap_thr_sec: float = 0.0
    try:
        if labels and len(labels) == len(gaps_sec):
            unique = sorted(set(int(x) for x in labels))
            short_label: Optional[int] = unique[0] if len(unique) >= 1 else None
            if short_label is not None:
                short_gaps = [g for g, l in zip(gaps_sec, labels) if int(l) == int(short_label)]
                if short_gaps:
                    gap_thr_sec = float(max(short_gaps))
    except Exception:
        gap_thr_sec = 0.0
    if gap_thr_sec <= 0:
        # fallback to median gap
        try:
            gap_thr_sec = float(np.median(gaps_sec)) if gaps_sec else 300.0
        except Exception:
            gap_thr_sec = 300.0

    # Step 2: local rule-based micro-clustering on uncovered messages
    leftover_idx = [idx for idx in df.index.tolist() if idx not in covered_idx]
    leftover_idx.sort()
    sessions_micro: List[Dict[str, Any]] = []

    def flush_session(msg_indices: List[int]):
        if not msg_indices:
            return
        part = df.loc[msg_indices]
        center_time = part["CreateTime"].median()
        sessions_micro.append({
            "messages": [row for _, row in part.iterrows()],
            "start": df.at[msg_indices[0], "CreateTime"],
            "end": df.at[msg_indices[-1], "CreateTime"],
            "center": center_time,
        })

    curr: List[int] = []
    same_run = 0
    prev_idx_global: Optional[int] = None

    for idx in leftover_idx:
        # compute gap to previous global message
        if prev_idx_global is not None:
            delta = df.at[idx, "CreateTime"] - df.at[prev_idx_global, "CreateTime"]
            gap_ok = delta.total_seconds() <= gap_thr_sec and delta.total_seconds() > 0
        else:
            gap_ok = True

        sender_curr = str(df.at[idx, "sender"]) if "sender" in df.columns else str(df.at[idx, "display_sender"]) 
        sender_prev = None
        if prev_idx_global is not None:
            sender_prev = str(df.at[prev_idx_global, "sender"]) if "sender" in df.columns else str(df.at[prev_idx_global, "display_sender"]) 

        # decide start new round
        start_new = False
        if prev_idx_global is None:
            start_new = True
        else:
            if sender_prev is not None and sender_curr != sender_prev:
                start_new = True
            elif not gap_ok:
                start_new = True

        if start_new:
            # end previous round
            flush_session(curr)
            curr = [idx]
            same_run = 1
        else:
            # continuation
            if sender_prev == sender_curr:
                same_run += 1
                curr.append(idx)
                if same_run > int(consecutive_same_sender_limit):
                    # end current round due to too many consecutive same-sender
                    flush_session(curr)
                    curr = []
                    same_run = 0
            else:
                # sender changed -> end previous and start new
                flush_session(curr)
                curr = [idx]
                same_run = 1

        prev_idx_global = idx

    # flush tail
    flush_session(curr)

    # Combine major and micro sessions, sort by start time
    combined = (major_sessions + sessions_micro)
    combined.sort(key=lambda s: s.get("start"))
    info = {
        "status": "ok",
        "session_count": len(combined),
        "dbscan_eps_sec": int(eps_seconds),
        "dbscan_min_samples": int(min_samples),
        "gap_threshold_sec": float(gap_thr_sec),
        "consecutive_same_sender_limit": int(consecutive_same_sender_limit),
        "cluster_centers": [s.get("center") for s in combined],
    }
    return combined, info


def sessionize_coarse_then_dbscan(
    df: pd.DataFrame,
    coarse_gap_minutes: int = 30,
    eps_seconds: int = 120,
    min_samples: int = 2,
    short_gap_seconds: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Coarse time + fine density sessionization:
    1) Build coarse segments using large time gaps (>= coarse_gap_minutes).
    2) Within each coarse segment, apply 1D-DBSCAN with small eps to find sub-sessions.
    3) If a sub-session has <2 messages and is very close to its neighbor (gap < short_gap_seconds), force-merge.
    Returns sub-sessions list and info.
    """
    if df.empty:
        return [], {"status": "empty"}

    times = df["CreateTime"].tolist()
    n = len(times)
    if n == 1:
        single = {"messages": [row for _, row in df.iterrows()], "start": times[0], "end": times[0]}
        info = {
            "status": "single",
            "session_count": 1,
            "coarse_gap_minutes": coarse_gap_minutes,
            "dbscan_eps_sec": int(eps_seconds),
            "dbscan_min_samples": int(min_samples),
            "short_gap_sec": int(short_gap_seconds),
        }
        return [single], info

    # Step 1: coarse boundaries
    gaps_sec = compute_reply_gaps(df)
    boundaries: List[int] = []
    for i, g in enumerate(gaps_sec, start=1):
        if float(g) >= float(coarse_gap_minutes) * 60.0:
            boundaries.append(i)

    # Step 2: fine DBSCAN within each coarse segment
    sub_sessions: List[Dict[str, Any]] = []
    start_idx = 0
    for i in boundaries:
        part = df.iloc[start_idx:i]
        sessions_part, _ = sessionize_dbscan(
            part,
            eps_seconds=eps_seconds,
            min_samples=min_samples,
        )
        sub_sessions.extend(sessions_part)
        start_idx = i
    part = df.iloc[start_idx:n]
    sessions_part, _ = sessionize_dbscan(
        part,
        eps_seconds=eps_seconds,
        min_samples=min_samples,
    )
    sub_sessions.extend(sessions_part)

    # Step 3: merge tiny sessions if near neighbor
    sub_sessions.sort(key=lambda s: s.get("start"))
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(sub_sessions):
            cur = sub_sessions[i]
            msgs = cur.get("messages", [])
            if len(msgs) < 2:
                prev_gap = float("inf")
                next_gap = float("inf")
                if i > 0:
                    prev_gap = (cur["start"] - sub_sessions[i - 1]["end"]).total_seconds()
                if i < len(sub_sessions) - 1:
                    next_gap = (sub_sessions[i + 1]["start"] - cur["end"]).total_seconds()
                min_gap = min(prev_gap, next_gap)
                if min_gap < float(short_gap_seconds):
                    if prev_gap <= next_gap and i > 0:
                        sub_sessions[i - 1]["messages"].extend(msgs)
                        sub_sessions[i - 1]["end"] = max(sub_sessions[i - 1]["end"], cur["end"])
                        del sub_sessions[i]
                        changed = True
                        continue
                    elif i < len(sub_sessions) - 1:
                        sub_sessions[i + 1]["messages"] = msgs + sub_sessions[i + 1]["messages"]
                        sub_sessions[i + 1]["start"] = min(cur["start"], sub_sessions[i + 1]["start"])
                        del sub_sessions[i]
                        changed = True
                        continue
            i += 1

    info = {
        "status": "ok",
        "session_count": len(sub_sessions),
        "initial_coarse_boundaries": boundaries,
        "coarse_gap_minutes": int(coarse_gap_minutes),
        "dbscan_eps_sec": int(eps_seconds),
        "dbscan_min_samples": int(min_samples),
        "short_gap_sec": int(short_gap_seconds),
    }
    return sub_sessions, info


def sessionize_sliding_window(
    df: pd.DataFrame,
    window_minutes: int = 15,
    avg_gap_threshold_minutes: float = 2.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    滑动窗口算法：
    1) 定义固定宽度窗口（如 15 分钟），窗口内消息视为候选组；
    2) 计算窗口内连续时间间隔的平均值；
    3) 若窗口平均间隔 < 阈值且包含不同参与者（sender），判为对话开始；
    4) 向前扩展，直到条件不再满足，则结束当前轮。
    """
    if df.empty:
        return [], {"status": "empty"}
    # 输入数据在 prepare_dataframe 中已按 CreateTime 排序；此处避免再次排序以降低开销
    df2 = df.reset_index(drop=True)
    n = len(df2)
    sessions: List[Dict[str, Any]] = []
    session_bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    bound_indices: List[int] = []
    cluster_centers: List[pd.Timestamp] = []
    session_types: List[str] = []
    i = 0
    while i < n:
        start_time = pd.Timestamp(df2.iloc[i]["CreateTime"])
        window_end = start_time + pd.Timedelta(minutes=int(window_minutes))
        j = i
        while j < n and pd.Timestamp(df2.iloc[j]["CreateTime"]) <= window_end:
            j += 1
        # 候选组是 [i, j-1]
        if j - i >= 2:
            secs = [float(pd.Timestamp(df2.iloc[k]["CreateTime"]).timestamp()) for k in range(i, j)]
            diffs = [secs[k] - secs[k-1] for k in range(1, len(secs))]
            avg_gap_sec = float(np.mean(diffs)) if diffs else float("inf")
            sender_col = "sender" if "sender" in df2.columns else ("display_sender" if "display_sender" in df2.columns else None)
            participants = set(df2.iloc[i:j][sender_col].tolist()) if sender_col else set()
            if avg_gap_sec < float(avg_gap_threshold_minutes) * 60.0 and len(participants) >= 2:
                end = j - 1
                k = end + 1
                while k < n:
                    t_k = pd.Timestamp(df2.iloc[k]["CreateTime"])
                    if t_k > start_time + pd.Timedelta(minutes=int(window_minutes)):
                        break
                    # 尝试扩展一个点
                    secs.append(float(t_k.timestamp()))
                    diffs2 = [secs[m] - secs[m-1] for m in range(1, len(secs))]
                    avg_gap_sec2 = float(np.mean(diffs2)) if diffs2 else float("inf")
                    participants2 = set(df2.iloc[i:k+1][sender_col].tolist()) if sender_col else set()
                    if avg_gap_sec2 < float(avg_gap_threshold_minutes) * 60.0 and len(participants2) >= 2:
                        end = k
                        k += 1
                    else:
                        # 回退扩展尝试
                        secs.pop()
                        break
                subset = df2.iloc[i:end+1]
                sessions.append({
                    "messages": [row for _, row in subset.iterrows()],
                    "start": subset.iloc[0]["CreateTime"],
                    "end": subset.iloc[-1]["CreateTime"],
                })
                session_bounds.append((pd.Timestamp(subset.iloc[0]["CreateTime"]), pd.Timestamp(subset.iloc[-1]["CreateTime"])))
                if i > 0:
                    bound_indices.append(int(i))
                center_t = pd.Timestamp(subset.iloc[0]["CreateTime"]) + (pd.Timestamp(subset.iloc[-1]["CreateTime"]) - pd.Timestamp(subset.iloc[0]["CreateTime"])) / 2
                cluster_centers.append(center_t)
                session_types.append("双人互动")
                i = end + 1
                continue
        i += 1
    info = {
        "status": "sliding_window",
        "session_count": len(sessions),
        "session_bounds": session_bounds,
        "bound_indices": bound_indices,
        "cluster_centers": cluster_centers,
        "session_types": session_types,
    }
    return sessions, info

def sessionize_adaptive_iqr(
    df: pd.DataFrame,
    gap_multiplier: float = 1.5,
    gap_sensitivity: float = 1.8,
    window_min: int = 5,
    window_max: int = 50,
    merge_factor: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    自适应滑动窗口统计（单遍）会话切分算法：
    - 升序遍历消息，计算相邻间隔 Δt。
    - 维护“滑动窗口”上的局部中位数(local_median)与标准差(local_std)。窗口长度随密度动态变化（介于 window_min～window_max 条间隔）。
    - 分界判断：若当前间隔 `Δt_i > local_median + k × local_std`（k=gap_sensitivity），则认定为分界点；
    - 段定义：分界点 i 关闭上一段于消息 i，下一段从消息 i+1 开始；首段从第一条消息开始，末段到最后一条消息结束；
    - 段合并：若相邻两段的分界间隔（两段之间的空隙）< 全局中位间隔 × merge_factor，则合并为一段；
    - 段类型：段内同时出现两个不同 is_sender → “双人互动”，否则 → “单人独白”。
    - 返回 sessions（含 bounds/start/end/type/center）和 info（含全局中位数/均值、gap_sensitivity/window_min/window_max/merge_factor、边界索引、会话段列表）。
    """
    if df.empty:
        return [], {"status": "empty"}
    # df 已在 prepare_dataframe 中按 CreateTime 排序，这里仅重置索引以避免重复排序的 O(N logN) 开销
    df2 = df.reset_index(drop=True)
    times: List[pd.Timestamp] = [pd.Timestamp(t) for t in df2["CreateTime"].tolist()]
    n = len(times)
    if n == 1:
        sess = {
            "messages": [row for _, row in df2.iterrows()],
            "start": times[0],
            "end": times[0],
            "type": "单人独白",
            "center": times[0],
        }
        return [sess], {
            "status": "single", "session_count": 1,
            "global_median_gap_sec": 0.0, "global_mean_gap_sec": 0.0,
            "gap_factor": float(gap_multiplier),
            "session_bounds": [(times[0], times[0])], "session_types": ["单人独白"], "bound_indices": [],
            "forced_split_count": 0,
            "conversations": [{"start_time": times[0], "end_time": times[0], "duration": 0.0, "message_count": 1}],
        }

    # 相邻间隔（秒）
    gaps_sec = compute_reply_gaps(df2)
    global_median = float(np.median(gaps_sec)) if gaps_sec else 0.0
    global_mean = float(np.mean(gaps_sec)) if gaps_sec else 0.0

    # 段类型识别前缀（O(1)判断）
    sender_col_ok = "is_sender" in df2.columns
    if sender_col_ok:
        try:
            sender_arr = pd.to_numeric(df2["is_sender"], errors="coerce").fillna(0).astype(int).to_numpy()
        except Exception:
            sender_arr = df2["is_sender"].astype(int).to_numpy()
        is_me = (sender_arr == 1).astype(np.int64)
        is_you = (sender_arr == 0).astype(np.int64)
        prefix_me = np.cumsum(is_me)
        prefix_you = np.cumsum(is_you)

        def _seg_type_fast(L: int, R: int) -> str:
            left_me = int(prefix_me[L - 1]) if L > 0 else 0
            left_you = int(prefix_you[L - 1]) if L > 0 else 0
            me_cnt = int(prefix_me[R] - left_me)
            you_cnt = int(prefix_you[R] - left_you)
            return "双人互动" if (me_cnt > 0 and you_cnt > 0) else "单人独白"
    else:
        def _seg_type_fast(L: int, R: int) -> str:
            col = "sender" if "sender" in df2.columns else ("display_sender" if "display_sender" in df2.columns else None)
            vals = set(str(df2.iloc[k][col]) for k in range(L, R + 1)) if col else set()
            return "双人互动" if len(vals) >= 2 else "单人独白"

    # 自适应滑动窗口切分：基于 local_median + k*local_std 判断分界
    from collections import deque
    from bisect import bisect_left, insort
    import math

    def clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, val))

    eps = 1e-6
    def window_for_gap(g: float) -> int:
        # dt 越小（密度高），窗口越大；dt 越大（密度低），窗口越小
        ratio = (global_median / (g + eps)) if global_median > 0 else 1.0
        r = math.sqrt(max(0.0, ratio))
        r = max(0.0, min(1.0, r))
        w = int(round(float(window_min) + (float(window_max - window_min) * r)))
        return clamp(w, int(window_min), int(window_max))

    # 滑动窗口容器
    win_vals: deque = deque()
    sorted_vals: List[float] = []
    sum_vals: float = 0.0
    sum_sqs: float = 0.0
    def add_val(v: float):
        insort(sorted_vals, v)
        win_vals.append(v)
        nonlocal sum_vals, sum_sqs
        sum_vals += v
        sum_sqs += v * v
    def remove_old():
        nonlocal sum_vals, sum_sqs
        old = win_vals.popleft()
        sum_vals -= old
        sum_sqs -= old * old
        idx = bisect_left(sorted_vals, old)
        # 处理可能存在的浮点近似，定位第一个等值元素
        while idx < len(sorted_vals) and sorted_vals[idx] != old:
            idx += 1
        if idx < len(sorted_vals):
            sorted_vals.pop(idx)

    def local_median() -> float:
        m = len(sorted_vals)
        if m == 0:
            return 0.0
        mid = m // 2
        if (m % 2) == 1:
            return float(sorted_vals[mid])
        else:
            return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)

    def local_std() -> float:
        m = len(win_vals)
        if m <= 1:
            return 0.0
        mean = sum_vals / float(m)
        var = max(0.0, (sum_sqs / float(m)) - (mean * mean))
        return float(math.sqrt(var))

    segments: List[Tuple[int, int]] = []  # (start_idx_message, end_idx_message)
    start_idx = 0
    bound_indices: List[int] = []
    min_stats_count = max(3, int(window_min))

    # 遍历每个相邻间隔，使用“前一窗口”的统计来判断当前间隔是否分界
    # gaps_sec[i] 是消息 i 与 i-1 的间隔；若分界，则上一段结束于消息 i-1，下一段从消息 i 开始
    for i in range(1, n):
        g = float(gaps_sec[i - 1])
        W = window_for_gap(g)
        # 调整窗口到目标大小（不包含当前间隔 g）
        while len(win_vals) > W:
            remove_old()
        # 只有当窗口内有足够历史间隔时才进行分界判断
        is_boundary = False
        if len(win_vals) >= min_stats_count:
            med = local_median()
            std = local_std()
            thr = float(med + float(gap_sensitivity) * (std if std > 0.0 else 0.0))
            if g > thr:
                is_boundary = True

        if is_boundary:
            # 关闭上一段于消息 i-1，下一段从 i 开始
            segments.append((start_idx, i - 1))
            bound_indices.append(int(i))
            start_idx = i
            # 重置窗口（新段重新统计）
            win_vals.clear(); sorted_vals.clear(); sum_vals = 0.0; sum_sqs = 0.0
            # 注意：边界间隔 g 不纳入新段窗口统计
            continue
        # 未分界：纳入窗口，参与下一个间隔的统计
        add_val(g)

    # 尾段：结束于最后一条消息
    segments.append((start_idx, n - 1))

    # 相邻短空隙合并：若两段间空隙 < 全局中位间隔 × merge_factor，则合并
    merged: List[Tuple[int, int]] = []
    if segments:
        merged.append(segments[0])
        for k in range(1, len(segments)):
            prev_L, prev_R = merged[-1]
            cur_L, cur_R = segments[k]
            gap_between = (times[cur_L] - times[prev_R]).total_seconds()
            if gap_between < (global_median * float(merge_factor)):
                merged[-1] = (prev_L, cur_R)
            else:
                merged.append((cur_L, cur_R))
    else:
        merged = segments

    # 构建 sessions 与信息（避免逐行 Series 复制，保留索引边界用于后续快速计算）
    sessions: List[Dict[str, Any]] = []
    session_bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    session_types: List[str] = []
    centers: List[pd.Timestamp] = []
    refined_bound_indices: List[int] = []
    for idx, (L, R) in enumerate(merged):
        s_type = _seg_type_fast(L, R)
        start_t = times[L]
        end_t = times[R]
        # 会话中心改为“消息时间中位数”，更贴近实际密度中心
        try:
            sub_ct = pd.to_datetime(pd.Series([df2.iloc[k]["CreateTime"] for k in range(L, R + 1)]))
            center_t = pd.Timestamp(sub_ct.median()) if len(sub_ct) > 0 else (start_t + (end_t - start_t) / 2)
        except Exception:
            center_t = start_t + (end_t - start_t) / 2
        # 不再复制每条消息，仅记录索引边界供后续计算
        sessions.append({"messages": [], "start": start_t, "end": end_t, "type": s_type, "center": center_t, "bounds": (L, R)})
        session_bounds.append((start_t, end_t))
        session_types.append(s_type)
        centers.append(center_t)
        if idx > 0:
            refined_bound_indices.append(int(L))

    # conversations 列表（作为分段表格源数据）
    conversations: List[Dict[str, Any]] = []
    for (s_t, e_t), (L, R) in zip(session_bounds, merged):
        duration_sec = float((pd.Timestamp(e_t) - pd.Timestamp(s_t)).total_seconds())
        conversations.append({
            "start_time": pd.Timestamp(s_t),
            "end_time": pd.Timestamp(e_t),
            "duration": duration_sec,
            "message_count": int(R - L + 1),
        })

    info = {
        "status": "adaptive_iqr",  # 保持模式标识以兼容下游判断
        "session_count": len(sessions),
        "bound_indices": refined_bound_indices,
        "session_bounds": session_bounds,
        "session_types": session_types,
        "cluster_centers": centers,
        "global_median_gap_sec": global_median,
        "global_mean_gap_sec": global_mean,
        "gap_sensitivity": float(gap_sensitivity),
        "window_min": int(window_min),
        "window_max": int(window_max),
        "merge_factor": float(merge_factor),
        "center_mode": "median",
        "conversations": conversations,
    }
    return sessions, info

def sessionize_tslearn(
    df: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """tslearn 分段已移除：统一回退到自适应滑窗，保持现有功能不受影响。"""
    sessions_fallback, info_fallback = sessionize_adaptive_iqr(
        df,
        gap_multiplier=1.0,
        gap_sensitivity=float(kwargs.get("fallback_gap_sensitivity", 1.8)),
        window_min=int(kwargs.get("fallback_window_min", 5)),
        window_max=int(kwargs.get("fallback_window_max", 50)),
        merge_factor=float(kwargs.get("fallback_merge_factor", 0.5)),
    )
    info_fallback.update({"status": "tslearn_removed"})
    return sessions_fallback, info_fallback
def dialogue_metrics(
    df: pd.DataFrame,
    sessionize_mode: str = "auto",
    idle_minutes: int = 30,
    max_reply_hours: int = 24,
    hybrid_max_gap_minutes: int = 30,
    hybrid_window_minutes: int = 10,
    hybrid_density_quantile: float = 0.25,
    hybrid_density_drop_ratio: float = 0.3,
    sliding_window_minutes: int = 15,
    sliding_avg_gap_minutes: float = 2.0,
    iqr_gap_multiplier: float = 1.5,
    # 新增：自适应滑窗算法参数
    adaptive_gap_sensitivity: float = 1.8,
    adaptive_window_min: int = 5,
    adaptive_window_max: int = 50,
    adaptive_merge_factor: float = 0.5,
    # 新增：tslearn 聚类参数
    tslearn_n_clusters: int = 3,
    tslearn_metric: str = "euclidean",
    tslearn_coarse_factor: float = 1.5,
    dbscan_eps_seconds: int = 300,
    dbscan_min_samples: int = 3,
    coarse_short_gap_seconds: int = 60,
) -> Dict[str, Any]:
    # helper: enforce minimal session standard (>=2 messages and both participants)
    def _enforce_two_party_min(sessions_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        buffer: Optional[Dict[str, Any]] = None
        def valid(sess: Dict[str, Any]) -> bool:
            msgs = sess.get("messages", [])
            if len(msgs) < 2:
                return False
            senders = set(str(m["sender"]) for m in msgs)
            return len(senders) >= 2
        def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            msgs = a.get("messages", []) + b.get("messages", [])
            start = a.get("start")
            end = b.get("end")
            center_time = pd.Series([m["CreateTime"] for m in msgs]).median()
            return {"messages": msgs, "start": start, "end": end, "center": center_time}
        for s in sessions_list:
            if buffer is None:
                if valid(s):
                    out.append(s)
                else:
                    buffer = s
            else:
                buffer = merge(buffer, s)
                if valid(buffer):
                    out.append(buffer)
                    buffer = None
        if buffer is not None:
            # try merge tail buffer into previous if exists
            if out:
                tail = merge(out[-1], buffer)
                if valid(tail):
                    out[-1] = tail
                # else drop tail buffer
            # else drop buffer
        # recompute centers for all
        for s in out:
            if "center" not in s:
                s["center"] = pd.Series([m["CreateTime"] for m in s.get("messages", [])]).median()
        # derive bound indices by session starts (excluding first)
        bound_indices: List[int] = []
        for idx in range(1, len(out)):
            if out[idx]["messages"]:
                try:
                    start_row = out[idx]["messages"][0]
                    # Series from iterrows has .name as original index
                    bidx = int(start_row.name) if hasattr(start_row, "name") else None
                    if bidx is not None:
                        bound_indices.append(bidx)
                except Exception:
                    continue
        info = {"refined_min_two_party": True, "bound_indices": bound_indices, "cluster_centers": [s.get("center") for s in out]}
        return out, info

    # run sessionization per mode
    if sessionize_mode == "auto":
        sessions_raw, cluster_info = sessionize_by_clustering(df)
    elif sessionize_mode == "hybrid":
        sessions_raw, cluster_info = sessionize_hybrid(
            df,
            max_gap_minutes=hybrid_max_gap_minutes,
            window_minutes=hybrid_window_minutes,
            density_quantile=hybrid_density_quantile,
            density_drop_ratio=hybrid_density_drop_ratio,
        )
    elif sessionize_mode == "density_clusters":
        sessions_raw, cluster_info = sessionize_density_clusters(
            df,
            window_minutes=hybrid_window_minutes,
            density_quantile=hybrid_density_quantile,
        )
    elif sessionize_mode == "sliding":
        sessions_raw, cluster_info = sessionize_sliding_window(
            df,
            window_minutes=sliding_window_minutes,
            avg_gap_threshold_minutes=sliding_avg_gap_minutes,
        )
    elif sessionize_mode == "adaptive_iqr":
        sessions_raw, cluster_info = sessionize_adaptive_iqr(
            df,
            gap_multiplier=iqr_gap_multiplier,
            gap_sensitivity=adaptive_gap_sensitivity,
            window_min=int(adaptive_window_min),
            window_max=int(adaptive_window_max),
            merge_factor=adaptive_merge_factor,
        )
    elif sessionize_mode == "tslearn":
        sessions_raw, cluster_info = sessionize_tslearn(
            df,
            n_clusters=int(tslearn_n_clusters),
            metric=str(tslearn_metric),
            merge_factor=float(adaptive_merge_factor),
            ts_coarse_factor=float(tslearn_coarse_factor) if 'tslearn_coarse_factor' in locals() else 1.5,
            fallback_gap_sensitivity=float(adaptive_gap_sensitivity),
            fallback_window_min=int(adaptive_window_min),
            fallback_window_max=int(adaptive_window_max),
            fallback_merge_factor=float(adaptive_merge_factor),
        )
    elif sessionize_mode == "dbscan":
        sessions_raw, cluster_info = sessionize_dbscan(
            df,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
        )
    elif sessionize_mode == "dbscan_rule":
        sessions_raw, cluster_info = sessionize_dbscan_rule(
            df,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
            consecutive_same_sender_limit=2,
        )
    elif sessionize_mode == "coarse_dbscan":
        sessions_raw, cluster_info = sessionize_coarse_then_dbscan(
            df,
            coarse_gap_minutes=hybrid_max_gap_minutes,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
            short_gap_seconds=coarse_short_gap_seconds,
        )
    else:
        sessions_raw = sessionize(df, idle_minutes=idle_minutes)
        cluster_info = {"status": "manual", "session_count": len(sessions_raw)}
    if not sessions_raw:
        return {}

    # record initial session count (raw)
    initial_count = len(sessions_raw)
    cluster_info.update({"initial_session_count": initial_count})
    # enforce最小双人标准：滑窗/tslearn/混合模式下直接使用原始分段（保留单人独白段）
    if sessionize_mode in ("hybrid", "sliding", "adaptive_iqr", "tslearn"):
        sessions = sessions_raw
        # 补充中心
        for s in sessions:
            if "center" not in s:
                s["center"] = pd.Series([m["CreateTime"] for m in s.get("messages", [])]).median()
        # 更新计数与中心
        cluster_info.update({
            "refined_min_two_party": False,
            "cluster_centers": [s.get("center") for s in sessions],
            "session_count": len(sessions),
        })
    else:
        sessions, refine_info = _enforce_two_party_min(sessions_raw)
        # merge refine info into cluster_info, recompute session_count
        cluster_info.update(refine_info)
        cluster_info["session_count"] = len(sessions)

    # Initiator stats（优先使用索引边界，避免逐条读取 messages）
    if sessionize_mode in ("adaptive_iqr", "tslearn"):
        initiators = []
        for s in sessions:
            b = s.get("bounds")
            if b and isinstance(b, tuple):
                L = int(b[0])
                try:
                    initiators.append(df.iloc[L]["sender"])
                except Exception:
                    # 回退：若索引异常则尝试 messages
                    if s.get("messages"):
                        initiators.append(s["messages"][0]["sender"]) 
            else:
                # 回退：使用 messages
                if s.get("messages"):
                    initiators.append(s["messages"][0]["sender"]) 
    else:
        initiators = [s["messages"][0]["sender"] for s in sessions]
    initiator_counts = Counter(initiators)

    # Average intra-session message gap（用边界索引进行向量化差分）
    gap_secs: List[float] = []
    if sessionize_mode in ("adaptive_iqr", "tslearn"):
        for s in sessions:
            b = s.get("bounds")
            if b and isinstance(b, tuple):
                L, R = int(b[0]), int(b[1])
                if R > L:
                    try:
                        sub = df.iloc[L:R+1]
                        diffs_sec = sub["CreateTime"].diff().iloc[1:].dt.total_seconds()
                        # 仅统计正间隔
                        gap_secs.extend([float(x) for x in diffs_sec.values if x and x > 0])
                    except Exception:
                        pass
    else:
        gaps = []
        for s in sessions:
            msgs = s.get("messages", [])
            for i in range(1, len(msgs)):
                gaps.append(msgs[i]["CreateTime"] - msgs[i - 1]["CreateTime"])
        gap_secs = [g.total_seconds() for g in gaps if g.total_seconds() > 0]
    avg_gap = float(np.mean(gap_secs)) if gap_secs else float("nan")

    # Average inter-session gap (between consecutive sessions)
    inter_gap_secs = []
    for i in range(1, len(sessions)):
        g = sessions[i]["start"] - sessions[i - 1]["end"]
        if g.total_seconds() > 0:
            inter_gap_secs.append(g.total_seconds())
    avg_inter_gap = float(np.mean(inter_gap_secs)) if inter_gap_secs else float("nan")

    # Link share stats
    link_df = df[df["is_link"]]
    link_counts_by_sender = link_df["sender"].value_counts().to_dict()

    # Swift reply threshold from clustering of all consecutive gaps
    gaps_all = compute_reply_gaps(df)
    cl = cluster_gaps(gaps_all)
    labels = cl.get("labels", [])
    swift_threshold_sec = float("nan")
    try:
        if labels and len(labels) == len(gaps_all):
            short_gaps = [g for g, l in zip(gaps_all, labels) if l == 0]
            if short_gaps:
                swift_threshold_sec = float(max(short_gaps))
    except Exception:
        swift_threshold_sec = float("nan")

    # After a link share, probability that next other-side message is 'swift' (<= short-cluster max)
    max_delta = timedelta(hours=max_reply_hours)
    def swift_reply_prob(from_label: str, to_label: str) -> Tuple[float, int]:
        """Swift reply probability = successes / total_links.
        Successes: link messages whose FIRST other-side reply appears within swift_threshold_sec.
        Total_links: COUNT of link messages sent by from_label (regardless of reply).
        """
        cand = link_df[link_df["sender"] == from_label]
        successes = 0
        total = int(cand.shape[0])
        for i in cand.index.tolist():
            t0 = df.at[i, "CreateTime"]
            # scan forward until max_delta; find the first other-side message
            for j in range(i + 1, len(df)):
                t1 = df.at[j, "CreateTime"]
                delta = t1 - t0
                if delta <= timedelta(0):
                    continue
                if delta > max_delta:
                    break
                if str(df.at[j, "sender"]) == to_label:
                    # first other-side reply within max window
                    if not np.isnan(swift_threshold_sec) and (delta.total_seconds() <= swift_threshold_sec):
                        successes += 1
                    break
            # if no other-side reply within window, this link is excluded from denominator
        prob = (successes / total) if total > 0 else float("nan")
        return (prob, total)

    me_prob, me_total = swift_reply_prob("我", "对方")
    you_prob, you_total = swift_reply_prob("对方", "我")

    result = {
        "initiator_counts": initiator_counts,
        "avg_intra_gap_sec": avg_gap,
        "avg_inter_session_gap_sec": avg_inter_gap,
        "link_counts_by_sender": link_counts_by_sender,
        "link_swift_reply_prob": {
            "me": {"prob": me_prob, "n": me_total},
            "you": {"prob": you_prob, "n": you_total},
        },
        "swift_threshold_sec": swift_threshold_sec,
        "session_count": len(sessions),
        "cluster_info": cluster_info,
        # conversations 作为顶层输出，方便表格直接使用
        "conversations": cluster_info.get("conversations", []),
    }
    return result


DEFAULT_STOPWORDS = set(
    [
        "的","了","和","是","我","你","他","她","它","在","就","都","而","及","与","并","或","被","把","等","于","上","下","中","这","那","也","很","还","又","呢","嘛","啊","哦","嗯","吧","呀","呗","呢","啊","吧","了","呢","吗","呀","哦",
    ]
)


def sentiment_and_emoji(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    text_df = df[df["type_norm"].isin(["文本", "引用回复"])]
    sentiments = []
    if SnowNLP is not None:
        for msg in text_df["msg"].tolist():
            try:
                s = SnowNLP(msg)
                sentiments.append(s.sentiments)  # 0-1 (越大越积极)
            except Exception:
                continue
    sentiment_stats = {
        "avg": float(np.mean(sentiments)) if sentiments else float("nan"),
        "median": float(np.median(sentiments)) if sentiments else float("nan"),
        "n": len(sentiments),
    }

    # word frequency via jieba
    top_words = []
    if jieba is not None and not text_df.empty:
        freq = Counter()
        for msg in text_df["msg"].tolist():
            try:
                for w in jieba.cut(msg):
                    w = w.strip()
                    if not w:
                        continue
                    if w in DEFAULT_STOPWORDS:
                        continue
                    # skip punctuation
                    if re.fullmatch(r"\W+", w):
                        continue
                    freq[w] += 1
            except Exception:
                continue
        top_words = freq.most_common(30)

    # emoji frequency
    emoji_counter = Counter()
    for ems in df["emojis"].tolist():
        for e in ems:
            emoji_counter[e] += 1
    top_emojis = emoji_counter.most_common(30)

    return {
        "sentiment": sentiment_stats,
        "top_words": top_words,
        "top_emojis": top_emojis,
    }


def shen_short_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    统计“神”短文本：当 type_norm ∈ {文本, 引用回复} 且去空白后的文本长度≤3，且包含“神”字。
    返回总数与明细（包含时间、日期、发送者、原文本）。
    """
    if df.empty:
        return {}
    try:
        text_df = df[df["type_norm"].isin(["文本", "引用回复"])].copy()
    except Exception:
        text_df = df.copy()

    def _is_shen_short(msg: Any) -> bool:
        try:
            s = str(msg)
            s2 = re.sub(r"\s+", "", s)
            return ("神" in s2) and (len(s2) <= 3)
        except Exception:
            return False

    try:
        cand = text_df[text_df["msg"].apply(_is_shen_short)].copy()
    except Exception:
        cand = pd.DataFrame(columns=text_df.columns)

    records: List[Dict[str, Any]] = []
    for idx, r in cand.iterrows():
        try:
            t = r["CreateTime"]
            sender_label = r["display_sender"] if "display_sender" in r and pd.notna(r["display_sender"]) else r.get("sender", "")
            records.append({
                "time": t,
                "date": str(getattr(t, "date", lambda: t)()) if hasattr(t, "date") else str(t),
                "sender": str(sender_label),
                "text": str(r.get("msg", "")),
            })
        except Exception:
            continue

    return {
        "count": int(len(records)),
        "records": records,
    }


def per_user_stats(df: pd.DataFrame, sender_col: str = "sender") -> Dict[str, Any]:
    if df.empty or sender_col not in df.columns:
        return {}
    # avg message length per user（按指定的发送者列分组）
    avg_len = df.groupby(sender_col)["message_length"].mean().to_dict()
    # type distribution per user（按指定的发送者列分组）
    type_dist = df.groupby([sender_col, "type_norm"]).size().reset_index(name="count")
    type_dist_pivot = type_dist.pivot(index="type_norm", columns=sender_col, values="count").fillna(0)
    return {
        "avg_len": avg_len,
        "type_dist": type_dist_pivot,
    }


def emoji_package_analysis(
    df: pd.DataFrame,
    sessionize_mode: str = "adaptive_iqr",
    # 兼容旧参数（保留但在 adaptive_iqr 模式下忽略）
    hybrid_max_gap_minutes: int = 30,
    hybrid_window_minutes: int = 10,
    hybrid_density_quantile: float = 0.25,
    sliding_window_minutes: int = 15,
    sliding_avg_gap_minutes: float = 2.0,
    # 新参数：与“对话模式”一致
    iqr_gap_multiplier: float = 1.0,
    gap_sensitivity: float = 1.8,
    window_min: int = 5,
    window_max: int = 50,
    merge_factor: float = 0.5,
    dbscan_eps_seconds: int = 300,
    dbscan_min_samples: int = 3,
) -> Dict[str, Any]:
    """
    分析动画表情（表情包）使用情况：
    1) 每轮会话表情包使用占比（每轮：动画表情消息数 / 会话总消息数）
    2) 一条消息后接动画表情的概率（整体与按发送方）
    3) 连续发送3个及以上动画表情的出现频率（总次数与每百条消息出现次数）
    """
    if df.empty:
        return {}
    # 若严格按原始类型，仅当 type_name=="动画表情" 存在时才进行统计
    if df[df["type_name"] == "动画表情"].empty:
        return {}

    # Sessionize: align with dialogue mode choices（默认 adaptive_iqr，与对话模式一致）
    if sessionize_mode == "hybrid":
        sessions, info = sessionize_hybrid(
            df,
            max_gap_minutes=hybrid_max_gap_minutes,
            window_minutes=hybrid_window_minutes,
            density_quantile=hybrid_density_quantile,
        )
    elif sessionize_mode == "density_clusters":
        sessions, info = sessionize_density_clusters(
            df,
            window_minutes=hybrid_window_minutes,
            density_quantile=hybrid_density_quantile,
        )
    elif sessionize_mode == "auto":
        sessions, info = sessionize_by_clustering(df)
    elif sessionize_mode == "sliding":
        sessions, info = sessionize_sliding_window(
            df,
            window_minutes=sliding_window_minutes,
            avg_gap_threshold_minutes=sliding_avg_gap_minutes,
        )
    elif sessionize_mode == "adaptive_iqr":
        sessions, info = sessionize_adaptive_iqr(
            df,
            gap_multiplier=iqr_gap_multiplier,
            gap_sensitivity=gap_sensitivity,
            window_min=int(window_min),
            window_max=int(window_max),
            merge_factor=merge_factor,
        )
    elif sessionize_mode == "dbscan":
        sessions, info = sessionize_dbscan(
            df,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
        )
    elif sessionize_mode == "dbscan_rule":
        sessions, info = sessionize_dbscan_rule(
            df,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
            consecutive_same_sender_limit=2,
        )
    elif sessionize_mode == "coarse_dbscan":
        sessions, info = sessionize_coarse_then_dbscan(
            df,
            coarse_gap_minutes=hybrid_max_gap_minutes,
            eps_seconds=dbscan_eps_seconds,
            min_samples=dbscan_min_samples,
            short_gap_seconds=60,
        )
    else:
        sessions = sessionize(df, idle_minutes=hybrid_max_gap_minutes)
        info = {"status": "manual", "session_count": len(sessions)}

    # 1) 每轮会话表情包占比
    session_props: List[float] = []
    for s in sessions:
        # 优先使用 bounds 以避免复制 messages 带来的性能开销
        bounds = s.get("bounds")
        if isinstance(bounds, tuple) and len(bounds) == 2:
            L, R = int(bounds[0]), int(bounds[1])
            subset = df.iloc[L:R+1]
            emoji_count = int((subset["type_name"] == "动画表情").sum())
            total = int(subset.shape[0])
            prop = float(emoji_count) / float(total) if total > 0 else float("nan")
            session_props.append(prop)
        else:
            msgs = s.get("messages", [])
            if not msgs:
                session_props.append(float("nan"))
                continue
            emoji_count = sum(1 for m in msgs if str(m.get("type_name")) == "动画表情")
            prop = float(emoji_count) / float(len(msgs)) if len(msgs) > 0 else float("nan")
            session_props.append(prop)
    props_clean = [p for p in session_props if not (isinstance(p, float) and np.isnan(p))]
    props_stats = {
        "mean": float(np.mean(props_clean)) if props_clean else float("nan"),
        "median": float(np.median(props_clean)) if props_clean else float("nan"),
        "n_sessions": len(session_props),
        "values": session_props,
    }

    # 2) 下一条是动画表情的概率（整体与按发送方）
    total_pairs = max(0, len(df) - 1)
    next_emoji = 0
    next_emoji_me = 0
    next_emoji_you = 0
    total_me = 0
    total_you = 0
    for i in range(total_pairs):
        cur_sender = df.iloc[i]["sender"]
        nxt_is_emoji = str(df.iloc[i + 1]["type_name"]) == "动画表情"
        if nxt_is_emoji:
            next_emoji += 1
        if cur_sender == "我":
            total_me += 1
            if nxt_is_emoji:
                next_emoji_me += 1
        else:
            total_you += 1
            if nxt_is_emoji:
                next_emoji_you += 1
    prob_overall = (next_emoji / total_pairs) if total_pairs > 0 else float("nan")
    prob_me = (next_emoji_me / total_me) if total_me > 0 else float("nan")
    prob_you = (next_emoji_you / total_you) if total_you > 0 else float("nan")

    next_probs = {
        "overall": prob_overall,
        "me": prob_me,
        "you": prob_you,
        "counts": {
            "pairs": total_pairs,
            "overall_emoji_next": next_emoji,
            "me_pairs": total_me,
            "me_emoji_next": next_emoji_me,
            "you_pairs": total_you,
            "you_emoji_next": next_emoji_you,
        },
    }

    # 3) 连续 >=3 动画表情的出现频率
    seq_count = 0
    cur_len = 0
    for i in range(len(df)):
        if str(df.iloc[i]["type_name"]) == "动画表情":
            cur_len += 1
        else:
            if cur_len >= 3:
                seq_count += 1
            cur_len = 0
    if cur_len >= 3:
        seq_count += 1
    total_msgs = len(df)
    rate_per_100 = (seq_count / total_msgs) * 100.0 if total_msgs > 0 else float("nan")
    seq_stats = {
        "count": int(seq_count),
        "rate_per_100_msgs": float(rate_per_100),
    }

    # 4) 表情包内容（src）频次排行
    pkg_df = df[df["type_name"] == "动画表情"].copy()
    def _src_id(v: Any) -> str:
        try:
            if v is None:
                return ""
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, dict):
                # 优先使用常见字段
                for key in ("url", "link", "src", "md5", "hash"):
                    if key in v and v[key]:
                        return str(v[key]).strip()
                return json.dumps(v, ensure_ascii=False)
            if isinstance(v, list):
                return json.dumps(v, ensure_ascii=False)
            return str(v)
        except Exception:
            return ""

    pkg_df["pkg_id"] = pkg_df["src"].apply(_src_id)
    pkg_counts = pkg_df["pkg_id"].value_counts()
    # 过滤空ID
    pkg_counts = pkg_counts[pkg_counts.index.astype(str).str.len() > 0]
    package_top = list(pkg_counts.items())[:30]

    return {
        "session_props": props_stats,
        "next_probs": next_probs,
        "seq_stats": seq_stats,
        "package_top": package_top,
        "sessionize_info": info,
    }
