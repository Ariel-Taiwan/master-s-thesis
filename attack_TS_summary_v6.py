# process_taxonomy.py
# 讀取輸入的 CSV，不限欄位，搜尋每一列中所有文字，擷取 "ENTRY ID:<數字>" 作為 ID，並計算 Likelihood Of Attack 和 Typical Severity 的數值總和，輸出結果至新的 CSV。

import pandas as pd
import re

# 參數：請將 input_path 設為來源 CSV 檔案路徑，output_path 設為輸出 CSV 檔案路徑
input_path = "658.csv"
output_path = "658_output.csv"

# 讀取 CSV，避免欄位名稱空白問題
df = pd.read_csv(input_path, engine="python", skipinitialspace=True)

# 定義ENTRY ID 的正則（忽略大小寫）
id_pattern = re.compile(r"TAXONOMY NAME:ATTACK:ENTRY ID:([\d\.]+)", re.IGNORECASE)

def extract_id_from_row(row):
    # 將整列所有值轉成字串後合併，再搜尋
    joined = " ".join(row.dropna().astype(str).tolist())
    m = id_pattern.search(joined)
    return m.group(1) if m else ""

# 應用 extract_id_from_row 來產生 ID
df["ID"] = df.apply(extract_id_from_row, axis=1)

# 定義 Likelihood/Severity 級別對應數值的字典
mapping = {"Very Low":1, "Low":2, "Medium":3, "High":4, "Very High":5}

# 將 Likelihood Of Attack / Typical Severity map 成數值，空值或非對應文字視為 0
df["Likelihood_num"] = df.get("Alternate Terms", pd.Series()).map(mapping).fillna(0).astype(int)
df["Severity_num"]   = df.get("Likelihood Of Attack", pd.Series()).map(mapping).fillna(0).astype(int)

# 計算總分 TS
df["TS"] = df["Likelihood_num"] + df["Severity_num"]

# 輸出：ID, TS, 原始 Likelihood, 原始 Severity
output_df = pd.DataFrame({
    "MITRE ID": df["ID"],
    "TS": df["TS"],
    "Likelihood": df.get("Alternate Terms", pd.Series()),
    "Severity": df.get("Likelihood Of Attack", pd.Series())
})

output_df.to_csv(output_path, index=False)
print(f"已生成：{output_path}")

