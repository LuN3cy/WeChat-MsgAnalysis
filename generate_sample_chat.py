import json
import random
from datetime import datetime, timedelta


def main():
    random.seed(42)

    total_days = 5
    # 不均匀的每日消息目标总量，合计300
    day_targets = [60, 50, 70, 40, 80]
    assert sum(day_targets) == 300

    # Base dates
    start_date = datetime(2024, 3, 10)
    # Time slots to cover 24h diversity
    slots = [
        (0, 30),   # 00:30
        (3, 30),   # 03:30
        (7, 0),    # 07:00
        (12, 15),  # 12:15
        (18, 0),   # 18:00
        (22, 30),  # 22:30
    ]

    phrases = [
        "早呀，今天有空吗？", "下午一起看展？", "晚上跑步？", "今天学习安排如何？", "这篇文章挺有意思",
        "周末去哪儿玩？", "昨天练得不错", "天气真好", "新的工具推荐看看", "今晚早点休息",
        "周一加油！", "活动报名链接", "已经报名了", "一起准备材料吧", "太好了！",
        "周末要好好休息", "走起～", "看起来很完美", "路线参考", "晚安～",
    ]

    emojis = ["😄", "😂", "👍", "💪", "👌", "🙂", "😎", "🥳", "🔥", "❤️"]
    image_desc = ["[图片] 早餐", "[图片] 风景", "[图片] 上班路上", "[图片] 晚餐", "[图片] 读书"]

    link_sources = [
        "https://example.com/exhibit",
        "https://news.example.com/article/123",
        "https://maps.example.com/route",
        "http://blog.example.com/post",
        "https://tools.example.com/",
        "https://docs.example.com/workout",
        "https://event.example.com/signup",
        "https://calendar.example.com/weekend",
        "https://video.example.com/highlights",
        "http://site.example.com/info",
    ]

    def pick_type():
        # weighted choice
        r = random.random()
        if r < 0.58:
            return "文本"
        elif r < 0.72:
            return "表情"
        elif r < 0.84:
            return "图片"
        elif r < 0.97:
            return "链接"
        elif r < 0.99:
            return "语音"
        else:
            return "视频"

    def make_msg(tname):
        if tname == "文本":
            base = random.choice(phrases)
            if random.random() < 0.3:
                base += random.choice(emojis)
            return base, ""
        elif tname == "表情":
            return random.choice(emojis), ""
        elif tname == "图片":
            return random.choice(image_desc), ""
        elif tname == "链接":
            url = random.choice(link_sources)
            msg = "看看这个链接" if random.random() < 0.5 else "参考一下"
            if random.random() < 0.5:
                src = {"url": url}
            else:
                src = url
            return msg, src
        elif tname == "语音":
            return "[语音] 片段", ""
        else:
            return "[视频] 片段", ""

    records = []
    id_base = 3000
    svr_base = 459005408925940000
    talker = "wxid_friend_123"

    per_day_actual = []

    for d in range(total_days):
        day_date = start_date + timedelta(days=d)
        target = day_targets[d]
        produced = 0
        # 直到当天达到目标条数
        while produced < target:
            # 任意选择一个时段作为会话起点（允许重复使用时段）
            h, m = random.choice(slots)
            current = datetime(day_date.year, day_date.month, day_date.day, h, m, 0)
            current += timedelta(seconds=random.randint(0, 180))  # 轻微抖动

            # 每个会话消息数在 [6, 14]，最后一轮按剩余量截断
            session_msgs = random.randint(6, 14)
            remaining = target - produced
            if session_msgs > remaining:
                session_msgs = remaining

            me_turn = random.random() < 0.5
            for _ in range(session_msgs):
                tname = pick_type()
                msg, src = make_msg(tname)

                is_sender = 1 if me_turn else 0
                rec = {
                    "id": id_base,
                    "MsgSvrID": str(svr_base + id_base),
                    "type_name": tname,
                    "is_sender": is_sender,
                    "talker": talker,
                    "room_name": talker,
                    "msg": msg,
                    "src": src,
                    "extra": {},
                    "CreateTime": current.strftime("%Y-%m-%d %H:%M:%S"),
                }
                records.append(rec)

                id_base += 1
                produced += 1

                # 典型会话内短间隔 + 偶发30-40分钟间隔
                gap_sec = random.randint(20, 120)
                if random.random() < 0.05:
                    gap_sec = random.randint(1800, 2400)
                current += timedelta(seconds=gap_sec)
                me_turn = not me_turn if random.random() > 0.25 else me_turn

        per_day_actual.append(produced)

    # sort by CreateTime for consistency
    records.sort(key=lambda r: r["CreateTime"]) 

    with open("sample_chat.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Daily counts:", per_day_actual)
    print(f"Generated {len(records)} messages across {total_days} days → sample_chat.json")


if __name__ == "__main__":
    main()
