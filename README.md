# 微信聊天数据分析与可视化（Python/Streamlit）

本项目用于对导出的微信聊天记录 JSON 数据进行本地分析与可视化，支持：

- 总览指标：双方消息占比、消息类型比例、时间分布热力图、周几最多、聊天时段分布（平均每日）
- 双方对比：平均消息长度、消息类型分布、平均响应时间（双向）
- 对话模式：会话分割与发起者统计、每轮对话平均间隔、链接分享统计、分享链接后被回复非链接消息的概率
- 内容分析：中文情感（SnowNLP）与表情/Emoji 使用频次统计

## 使用方法

1. 安装依赖：
   - 可使用虚拟环境（推荐）：
     - Windows PowerShell：
       ```powershell
       python -m venv .venv
       .\.venv\Scripts\python.exe -m pip install -r requirements.txt
       ```
   - 或直接安装到系统环境：
       ```powershell
       pip install -r requirements.txt
       ```

2. 启动可视化应用：
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
   启动后浏览器访问提示的 `http://localhost:8501`。

3. 在应用中上传你的 JSON 文件或指定本地路径，选择时间范围，查看各项分析。

## 数据格式说明

示例（来自你的导出）：
```json
{
  "id": 1875,
  "MsgSvrID": "459005408925941616",
  "type_name": "文本",
  "is_sender": 0,
  "talker": "wxid_vj38u8pa47t722",
  "room_name": "wxid_vj38u8pa47t722",
  "msg": "我觉得铜镜切入比九章算术好",
  "src": "",
  "extra": {},
  "CreateTime": "2024-02-28 16:52:37"
}
```
- `is_sender`：1 代表“我”，0 代表“对方”（默认约定，可在 UI 中切换）
- `type_name`：消息类型（如 文本/图片/表情/链接 等）
- `msg`：文本内容；`src`：富媒体/链接源（如不为空可视为链接）

## 说明与假设

- 会话分割阈值默认 30 分钟，可在 UI 调整。
- 响应时间统计会寻找对方的下一条消息，并做异常值裁剪（默认 24 小时）。
- 链接识别：`type_name` 包含“链接”或 `msg`/`src` 中包含 `http(s)`/`www` 时认定为链接。
- 中文情感分析基于 SnowNLP，仅对文本消息进行。
- Emoji 统计通过 unicode 提取（对“表情”图片类消息可能无法精准识别）。

## 依赖
- streamlit, pandas, numpy, plotly, seaborn, matplotlib
- jieba（中文分词）, snownlp（中文情感分析）, emoji（emoji 提取）

## 开发结构
- `app.py`：Streamlit 应用入口与页面布局
- `analysis.py`：数据处理与各类指标计算函数
- `requirements.txt`：依赖清单

如需扩展其他分析，请在 `analysis.py` 中新增相应函数，并在 `app.py` 中接入可视化。
