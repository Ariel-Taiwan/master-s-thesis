from collections import Counter

file_path = 'train_clean_escaped_save_before_added_cut_3042_add_4_escaped_removelong29.csv'  # 替換為你的實際檔案路徑
label_counter = Counter()

# 使用最基本的行讀取方式解析每一行
with open(file_path, 'r', encoding='utf-8') as f:
    next(f)  # 跳過標頭行
    for line in f:
        # 以最後一個逗號分開 text 和 labels 欄（避免 text 中的逗號干擾）
        if ',' not in line:
            continue
        parts = line.rsplit(',', 1)
        if len(parts) != 2:
            continue
        labels_str = parts[1].strip().strip('"')
        for label in labels_str.split(';'):
            label = label.strip()
            if label.isdigit():
                label_counter[label] += 1

# 顯示各個 label 的筆數（由小到大排序）
for label, count in sorted(label_counter.items(), key=lambda x: int(x[0])):
    print(f"Label {label}: {count}")

# 額外統計
print("\n最大筆數：", max(label_counter.values()))
print("最小筆數：", min(label_counter.values()))

