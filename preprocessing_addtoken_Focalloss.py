import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertConfig, AutoTokenizer, AutoModel, EarlyStoppingCallback, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Value
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
from imblearn.under_sampling import RandomUnderSampler
import random
import re
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

label2id = {
    "TA0040 - Impact/ T1490-Inhibit System Recovery": 0,
    "T1543.003 - Create or Modify System Process: Windows Service": 1,
    "T1082 - System Information Discovery": 2,
    "T1057 - Process Discovery": 3,
    "TA0007 - Discovery / T1069-Permission Groups Discovery": 4,
    "T1090.001 – Proxy: Internal Proxy": 5,
    "T1059.003 - Command and Scripting Interpreter: Windows Command Shell": 6,
    "T1059.001 - Command and Scripting Interpreter: PowerShell": 7,
    "T1021.004 - Remote Services: SSH": 8,
    "T1087 - Account Discovery": 9,
    "T1218.011 System Binary Proxy Execution: Rundll32": 10,
    "T1204.002 - User Execution: Malicious File": 11,
    "T1105 - Ingress Tool Transfer": 12,
    "T1071.001 - Application Layer Protocol: Web Protocols": 13,
    "T1055 – Process Injection": 14,
    "T1059.005 - Command and Scripting Interpreter: Visual Basic": 15,
    "T1018 - Remote System Discovery": 16,
    "T1218.014 - System Binary Proxy Execution: MMC": 17,
    "T1216.001 - Signed Binary Proxy Execution: cscript.exe": 18,
    "T1548.002 - Abuse Elevation Control Mechanism: Bypass User Account Control": 19,
    "T1040 - Network Sniffing": 20,
    "TA0008 - Lateral Movement / T1021.002-SMB Windows Admin Shares": 21,
    "T1003.001 - OS Credential Dumping: LSASS Memory": 22,
    "T1047 - Windows Management Instrumentation": 23,
    "TA0007 - Discovery / T1201-Password Policy Discovery": 24,
    "T1033 - System Owner/User Discovery": 25,
    "T1059.006 - Command and Scripting Interpreter: Python": 26,
    "TA0007 - Discovery / T1016-System Network Configuration Discovery": 27,
    "TA0002 - Execution/ T1053.005-Scheduled Task": 28,
    "TA0002 - Execution/ T1059.001-PowerShell": 29,
    "TA0002 - Execution/ T1059.003-Windows Command Shell": 30,
    "TA0002 - Execution/ T1204-User execution": 31,
    "TA0003 - Persistence/ T1098-Account manipulation": 32,
    "TA0003 - Persistence/ T1136-Create account": 33,
    "TA0003 - Persistence/ T1197-BITS jobs": 34,
    "TA0003 - Persistence/ T1505.001-SQL Stored Procedures": 35,
    "TA0003 - Persistence/ T1543.003-Create or Modify System Process-Windows Service": 36,
    "TA0003 - Persistence/ T1546-Event Triggered Execution": 37,
    "TA0007 - Discovery / T1135-Network Share Discovery": 38,
    "TA0004 - Privilege Escalation/ T1134-Access Token Manipulation": 39,
    "TA0004 - Privilege Escalation/ T1546-Image File Execution Options Injection": 40,
    "TA0004 - Privilege Escalation/ T1574-DLL side-loading": 41,
    "TA0005 - Defense Evasion/ T1027-Obfuscated Files or Information": 42,
    "TA0005 - Defense Evasion/ T1070.001-Clear Windows event logs": 43,
    "T1563.002 – Remote Service Session Hijacking: RDP Hijacking": 44,
    "TA0005 - Defense Evasion/ T1140-Deobfuscate-Decode Files or Information": 45,
    "TA0005 - Defense Evasion/ T1562.001-Impair Defenses-Disable or Modify tool": 46,
    "TA0005 - Defense Evasion/ T1562.002-Disable Windows Event Logging": 47,
    "TA0005 - Defense Evasion/ T1562.004-Impair Defenses-Disable or Modify System Firewall": 48,
    "TA0005 - Defense Evasion/ T1564-Hide artifacts": 49,
    "TA0006 - Credential Access/ T1003-Credential dumping": 50,
    "TA0006 - Credential Access/ T1040-Traffic sniffing": 51,
    "TA0008 - Lateral Movement / T1021.001-Remote Desktop Protocol": 52
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# 1. 定義 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        alpha: 正例權重（可設 <1 減少負例懲罰、>1 增加正例權重）
        gamma: 焦慮參數，越大越集中在難分類樣本
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # 我們用 BCEWithLogits 來同時計算 sigmoid + BCE
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_labels)
        targets: float Tensor, same shape, 0/1 multi-hot
        """
        # 先算出 element-wise 的 BCE loss
        bce_loss = self.bce(logits, targets)

        # sigmoid 後的機率
        p = torch.sigmoid(logits)

        # p_t = p when y=1, or (1-p) when y=0
        p_t = p * targets + (1 - p) * (1 - targets)

        # focal weight
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 原始 tensor

# 2. 繼承 Trainer，覆寫 compute_loss
class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs 裡面一定要有 "labels"
        labels = inputs.pop("labels")
        # forward
        outputs = model(**inputs)
        logits = outputs.logits   # shape (batch_size, num_labels)
        # 計算 Focal Loss
        loss = self.focal_loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

def mask_dynamic(text):
    text = re.sub(r'[0-9]+(\.[0-9]+){3}', '<IP>', text)
    # 把十進位 PID-> <PID>
    text = re.sub(r'\b\d+\b', '<PID>', text)
    # 把十六進位位址-> <HEX>
    text = re.sub(r'0x[0-9A-Fa-f]+', '<HEX>', text)
    # 把 GUID 形式 {…}-> <GUID>
    text = re.sub(r'\{[0-9A-Fa-f\-]+\}', '<GUID>', text)
    return text

def split_steps(text):
    steps = text.split('->')
    # 前後 strip，然後再 join 回一行並以特殊 token 分隔
    return ' [SEP] '.join(s.strip() for s in steps)

def split_camel_kebab(tok):
    tok = re.sub(r'([a-z])([A-Z])', r'\1 \2', tok)
    return tok.replace('-', ' - ')

def preprocess_powershell(text):
    # 1) 反转义
    text = text.replace('\\\\', '\\').replace('\\"', '"')
    # 2) 删除多余引号
    text = re.sub(r'"{2,}', '"', text)
    # 3) 移除 .strip()
    text = re.sub(r"\.strip\(\)", "", text)
    text = text.replace("\\", "/")   # 將反斜線統一為正斜線
    text = re.sub(r"/{2,}", "/", text)         # 把多個 // 合併成 /
    text = text.replace('"', " ")    # 去除雙引號但不改動路徑
    # 2) 把连续空格合并为一个
    text = re.sub(r"\s+", " ", text).strip()
    # 4) 统一分隔符
    text = text.replace("\\", "/")
    text = mask_dynamic(text)
    text = split_steps(text)
    #tokens = split_camel_kebab(text)     # 拆 CamelCase
    return text

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    # 多標籤 sigmoid
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()

    # micro
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0)
    # macro
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)

    # AUC (防呆：有正有負才算)
    auc = []
    for i in range(labels.shape[1]):
        if len(set(labels[:, i])) == 2:
            auc.append(roc_auc_score(labels[:, i], probs[:, i]))
    auc_macro = np.mean(auc) if auc else float("nan")

    return {
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": auc_macro
    }

model_name = "markusbayer/CySecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#tokens = tokenizer.tokenize("(C:\Windows\System32\cmd.exe c:\windows\system32\cmd.exe) (C:\Windows\System32\OpenSSH\ssh.exe ssh  user@140.123.105.150)")
#subword_count = sum(1 for t in tokens if t.startswith("##"))
#subword_ratio = subword_count / len(tokens)
#print(subword_ratio)

text = "Cherokee Webserver Latest Cherokee Web server Upto Version 1.2.103 (Current stable) is affected by: Buffer Overflow - CWE-120. The impact is: Crash. The component is: Main cherokee command. The attack vector is: Overwrite argv[0] to an insane length with execl. The fixed version is: There's no fix yet."

tokens = tokenizer.tokenize(text)
subword_count = sum(1 for t in tokens if t.startswith("##"))
subword_ratio = subword_count / len(tokens)
print(tokens)
print(subword_ratio)

dataset = load_dataset("csv",
    data_files={"train": "train_clean_escaped_save_before_added_cut_3042_add_4_escaped_removelong29_contextual_powershell_mmc.csv"},
    split="train"                  # <-- 這樣 dataset 就是一個 Dataset，不是 DatasetDict
)

# 應用到 dataset：
dataset = dataset.map(lambda ex: {"text": preprocess_powershell(ex["text"])})
#dataset = dataset.map(lambda ex: {"text": ex["text"]})

#4. 轉成 pandas DataFrame
#df = dataset.to_pandas()
# 5. 存成 CSV
#output_path = "preprocessed_powershell_removelong29.csv"
#df.to_csv(output_path, index=False, encoding="utf-8")
#print(f"已將處理後的資料匯出到 {output_path}")

# 為避免 CSV 自動將 labels 欄位轉為數值，這裡強制將其讀入為 string
dataset = dataset.cast_column("labels", Value("string"))

# 假设你已经做了 cast_column("labels", Value("string"))
print(dataset.column_names)
print(dataset["labels"][:10])

def compute_token_lengths(example):
    return {"token_length": len(tokenizer(example["text"])["input_ids"])}

token_lengths_dataset = dataset.map(compute_token_lengths)
token_lengths = np.array(token_lengths_dataset["token_length"])
print(f"最大 token 長度: {token_lengths.max()}")
print(f"平均 token 長度: {token_lengths.mean():.2f}")
print(f"90 百分位數: {np.percentile(token_lengths, 90)}")
print(f"95 百分位數: {np.percentile(token_lengths, 95)}")
print(f"99 百分位數: {np.percentile(token_lengths, 99)}")

# 定義一個轉換函數：將 CSV 中的 labels（如 "1" 或 "4;5"）轉換為長度為 num_labels 的 multi-hot 向量
def convert_labels(example):
    labels_str = example["labels"].strip().strip('"')
    # 若存在分號，表示多標籤；若無則只有單一標籤
    labels_list = labels_str.split(";")
    multi_hot = [0.0] * num_labels
    for lab in labels_list:
        lab = lab.strip()
        if lab.isdigit():
            # 假設 CSV 中的數字標籤為 1-index，故減 1 轉成 0-index
            idx = int(lab)
            if 0 <= idx < num_labels:
                multi_hot[idx] = 1.0
    example["labels"] = multi_hot
    return example

# 對資料集中的標籤進行轉換
dataset = dataset.map(convert_labels)

# —— 2. 定义正则，匹配 .exe 可执行文件 和 PowerShell cmdlet（Verb-Noun）
exe_pattern   = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\.exe\b")
pscmd_pattern = re.compile(r"\b[A-Za-z]+-[A-Za-z]+\b")

auto_tokens = set()
for text in dataset["text"]:
    if not text:
        continue
    auto_tokens.update(exe_pattern.findall(text))
    auto_tokens.update(pscmd_pattern.findall(text))
manual_tokens = {'sppsvc.exe', 'services.exe', 'slui.exe', 'Eng.exe', 'Mon.exe', 'ssh.exe', 'dllhost.exe', 'TSTheme.exe', 'NPFInstall.exe', 'whoami.exe', 'notepad.exe', 'cscript.exe', 'rundll32.exe', 'schtasks', 'msedge.exe', 'taskhostex.exe', 'taskmgr.exe', 'rdpinput.exe', 'forfiles.exe', 'ctfmon.exe', 'netsh.exe', 'PSEXESVC.exe', 'auditpol.exe', 'wbadmin.exe', 'atbroker.exe', 'Service.exe', 'payload.exe', '3virus.exe', 'plink.exe', 'Terminal.exe', 'net.exe', 'bitsadmin.exe', 'lsass.exe', 'conhost.exe', 'Taskmgr.exe', 'taskkill.exe', 'drvinst.exe', 'sethc.exe', 'Installer.exe', 'dumpcap.exe', 'Auth.exe', 'VSSVC.exe', 'SE.exe', 'tasklist.exe', 'python3.exe', 'nanodump.exe', 'Indexer.exe', 'mmc.exe', 'schtasks.exe', 'reg.exe', 'taskhostw.exe', 'netdom.exe', 'mimi.exe', 'winlogon.exe', 'wmic.exe', 'ntdsutil.exe', 'evil.exe', 'mimikatz.exe', 'tasklist', 'tscon.exe', 'timeout.exe', 'runas.exe', 'regedit.exe', 'injector.exe', 'powershell.exe', 'procdump.exe', 'Seclogon.exe', 'Broker.exe', 'Potato.exe', 'Dism.exe', 'wmiadap.exe', 'explorer.exe', 'pentestlab.exe', 'installer.exe', 'tshark.exe', 'Host.exe', 'TVupu.exe', 'py.exe', 'dump_lsass.exe', 'Inst.exe', 'WMIADAP.exe', 'diskshadow.exe', 'spoolsv.exe', 'vssvc.exe', 'consent.exe', 'wmiprvse.exe', 'certutil.exe', 'at.exe', 'setspn.exe', 'pdf.exe', 'wsl.exe', 'cmd.exe', 'svchost.exe', 'net1.exe', 'virus.exe', 'launchtm.exe', 'Worker.exe', 'rdpclip.exe', 'sspdumper.exe', 'myapp.exe', 'Wireshark.exe', 'sc.exe', 'sshd.exe', 'dnscmd.exe', 'UI0Detect.exe', 'wevtutil.exe', 'python.exe', 'dsquery.exe', 'TM.exe', 'mshta.exe', 'sqlservr.exe', 'WMIC.exe', 'dismhost.exe', 'whoami'}

#all_special = auto_tokens | manual_tokens
#special_tokens_list = sorted(all_special)
#print(f"一共准备添加 {len(special_tokens_list)} 个 special tokens，例如：")
#print(special_tokens_list)

special_tokens = ['powershell.exe']
tokenizer.add_special_tokens({
    'additional_special_tokens': special_tokens
})

# 定義 tokenize 函數：針對 text 欄位做 encoding，並設定 padding 與 truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True,max_length=256)

# 對資料集進行 tokenization（batched 模式）
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 設定格式轉換，將需要的欄位轉成 PyTorch tensor
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# X 部分：把 input_ids、attention_mask 取出來，轉成 NumPy
input_ids      = np.array(tokenized_datasets["input_ids"])      # shape (N, seq_len)
attention_mask = np.array(tokenized_datasets["attention_mask"]) # shape (N, seq_len)

# 通常我們把它們水平堆疊，做成一個 (N, 2*seq_len) 的矩陣
X = np.concatenate([input_ids, attention_mask], axis=1)         # shape (N, 2*seq_len)

# 準備 labels array for stratification
y = np.array(tokenized_datasets["labels"])  # shape = (N, num_labels)

# 拿到所有 multi-hot 向量
#labs = np.array(tokenized["labels"])  # 形状 (N, C)
print("Loaded dataset size:", y.shape)
print("Sum per class:", y.sum(axis=0))

def find_best_threshold(y_true_flat, y_scores_flat):
    prec, rec, th = precision_recall_curve(y_true_flat, y_scores_flat)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = np.argmax(f1[:-1])
    return th[best], f1[best]

def find_per_class_thresholds(y_true: np.ndarray, y_scores: np.ndarray):
    """
    對每個類別跑 precision–recall 曲線，找出讓該類別 F1 最大的 threshold。
    y_true: (N, C) 0/1 ground-truth
    y_scores: (N, C) [0,1] 預測機率
    回傳 length-C 的 best_thresholds array
    """
    C = y_true.shape[1]
    best_thresholds = np.zeros(C, dtype=float)
    for c in range(C):
        prec, rec, th = precision_recall_curve(y_true[:, c], y_scores[:, c])
        # precision_recall_curve 會返回 len(th)+1 長度的 prec/rec
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1[:-1])  # 忽略最後一個無對應 threshold 的點
        best_thresholds[c] = th[best_idx]
    return best_thresholds

def ensure_min_one_positive(val_idx, y, min_pos=1):
    """
    如果 val_idx 对应的 y[val_idx] 某些列全 0，
    就把训练集里（不在 val_idx）的一个正例 idx 加进 val_idx。
    """
    val_idx = set(val_idx)
    train_idx = set(range(len(y))) - val_idx
    val_labels = y[list(val_idx)]
    for c in range(y.shape[1]):
        if val_labels[:, c].sum() < min_pos:
            # 找训练集里第一个对该类为正的样本
            for i in train_idx:
                if y[i, c] == 1:
                    val_idx.add(i)
                    train_idx.remove(i)
                    break
    return list(train_idx), list(val_idx)

# 3. K 折初始化
K = 2
kf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=42)
fold_results = []

# y 是 (N, C) 的 multi-hot numpy array
total_counts = y.sum(axis=0)  # 每个标签在全数据里的正例数
print("All labels' total positive counts:\n", total_counts)

# 找出全数据里少于 K 的那些标签
labels_too_few = np.where(total_counts < K)[0]
print(f"Labels with fewer than {K} positives in TOTAL data:", labels_too_few)

for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y)), y), start=1):
    print(f"\n======== Fold {fold} / {K} ========")

    t_idx, v_idx = ensure_min_one_positive(val_idx, y)

    # 这里马上检查
    y_val = y[val_idx]
    pos_counts = y_val.sum(axis=0)   # 每个类在这个 val fold 的正例数
    print(f"Fold {fold} positive counts per class:", pos_counts)

    # 1. 拆 train / val
    train_sub = tokenized_datasets.select(t_idx)
    val_sub   = tokenized_datasets.select(v_idx)

    # 2. 手動對 train_sub 做欠採樣：Label 42 最多保留 20 筆
    #    a. 找到所有含 42 的樣本 index
    pos_idx = [i for i, lbls in enumerate(train_sub["labels"]) if 42 in lbls]
    #    b. 隨機抽 20 筆（若不足就全保留）
    keep_pos = pos_idx if len(pos_idx) <= 20 else random.sample(pos_idx, 20)
    #    c. 找到不含 42 的樣本 index
    neg_idx = [i for i in range(len(train_sub)) if i not in pos_idx]
    #    d. 合併後打散
    new_idx = keep_pos + neg_idx
    random.shuffle(new_idx)
    train_sub = train_sub.select(new_idx)

    # 3. 把 train_sub / val_sub 設成 PyTorch tensors 格式
    train_sub.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_sub.set_format(  "torch", columns=["input_ids", "attention_mask", "labels"])

    # 4b. 重置 model
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    model.resize_token_embeddings(len(tokenizer))
    print("新增後的詞表大小：", len(tokenizer))    
    #model.classifier = nn.Sequential(
      #  nn.Dropout(p=0.3),    # 新增这一行
     #   model.classifier      # 原来的 Linear(hidden_size → num_labels)
    #)
    
    #print(model.classifier[1].weight)
    #print(model.classifier[1].bias)
    #weight_before = model.classifier[1].weight.clone().detach()
    #bias_before = model.classifier[1].bias.clone().detach() 

    # 4c. TrainingArguments：每折都獨立一個 output_dir
    output_dir = "checkpoints_Six/full_model"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,# 可依折內資料量調整
        #warmup_ratio = 0.2, 
        weight_decay=0.01,
        metric_for_best_model="eval_f1_micro",  # 或 "eval_loss
        greater_is_better=True,
        load_best_model_at_end = True,
        save_total_limit=1,
        #logging_steps=50
    )

    # 4d. 建立 Trainer FocalLoss
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_sub,
        eval_dataset=val_sub,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        focal_alpha=0.9,
        focal_gamma=0.5,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    # 4e. train + eval
    trainer.train()
    metrics = trainer.evaluate()
    
    print(f"Fold {fold} metrics:", metrics)
    fold_results.append(metrics)
    
    import seaborn as sns

    # 1) 拿到該折的 raw logits & 真實 labels
    pred = trainer.predict(val_sub)
    logits_val = pred.predictions      # shape = (N_val, C)
    y_true     = np.array(val_sub["labels"])
    
    # 2. 計算機率
    y_scores_val = torch.sigmoid(torch.tensor(logits_val))
    
    y_pred = (y_scores_val > 0.5).int().numpy()
   
    # 4) 計算你要的各項指標
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) == 2:
            aucs.append(roc_auc_score(y_true[:, c], y_scores_val[:, c]))
    auc_macro = np.mean(aucs) if aucs else float("nan")

    fold_results.append({
        "fold": fold,
        #"best_th": best_th,
        #"f1_global": best_f1,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": auc_macro,
    })
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
n_classes = y_scores_val.shape[1]
y_test_bin = label_binarize(y_true, classes=range(n_classes))

# 2. 计算每个类别的 ROC 曲线和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores_val[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 3. 计算 “micro” 和 “macro” 平均曲线
# micro-average: 将所有类别的决策函数扁平化
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores_val.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro-average: 先汇总所有 fpr 点，再对 tpr 插值取平均
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 4. 画图
plt.figure(figsize=(8, 6))

# 对每个类别画一条线
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
#for i, color in zip(range(n_classes), colors):
 #   plt.plot(fpr[i], tpr[i], color=color, lw=1,
  #           label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

# 画 micro & macro
plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average (AUC = {roc_auc["micro"]:0.2f})',
         linestyle=':', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average (AUC = {roc_auc["macro"]:0.2f})',
         linestyle='--', linewidth=2)

# 对角线
plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 最後把所有 fold 的指标做i平均
import pandas as pd
df = pd.DataFrame(fold_results).set_index("fold")
print("\nCross-val results with dynamic threshold:")
print(df.mean())

# 你關心的幾個指標
metrics_data = ['eval_f1_micro', 'eval_f1_macro', 'eval_roc_auc_macro',
           'f1_micro', 'f1_macro', 'roc_auc_macro']

mean_scores = df.mean()[metrics_data]

# 畫圖
#plt.figure(figsize=(10, 5))
#mean_scores.plot(kind='bar', color='skyblue')
#plt.title('Cross-Validation Average Metrics (K-fold)')
#plt.ylabel('Score')
#plt.ylim(0, 1)
#plt.xticks(rotation=45)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.tight_layout()
#plt.show()

print(f"After per-class threshold:")
print(f"micro-F1 = {f1_micro:.4f}, macro-F1 = {f1_macro:.4f}, roc_auc_macro = {auc_macro:.4f}")

# 5. 把每折的指標平均
#keys = fold_results[0].keys()
#avg_results = {k: np.mean([m[k] for m in fold_results]) for k in keys}
#print("\n===== Cross-validation average results =====")
#for k, v in avg_results.items():
#    print(f"{k}: {v:.4f}")

# 訓練完成後，我們使用 Trainer 的 predict 方法來測試模型
# 定義一些測試文本
test_texts = [
    "(C:\\Windows\\System32\\cmd.exe) -> (C:\\Windows\\System32\\sc.exe sc.exe start testpayload)",
    "(powershell  -ExecutionPolicy Bypass -File Trigger_killchain.ps1) -> (C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -Command Start-Process cmd -ArgumentList '/c whoami /priv' -Verb RunAs ) -> (C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -Command Start-Process cmd -ArgumentList \"/c whoami /priv\" -Verb RunAs)",
    "(C:\Windows\System32\services.exe) -> (C:\Windows\System32\OpenSSH\sshd.exe C:\Windows\System32\OpenSSH\sshd.exe) -> (C:\Windows\System32\OpenSSH\sshd.exe ""C:\Windows\System32\OpenSSH\sshd.exe"" -R) (C:\Windows\System32\OpenSSH\sshd.exe ""C:\Windows\System32\OpenSSH\sshd.exe"" -z) (C:\Windows\System32\conhost.exe C:\Windows\system32\conhost.exe --headless --width 206 --height 58 --signal 0x1e0 -- ""c:\windows\system32\cmd.exe"") (C:\Windows\System32\cmd.exe c:\windows\system32\cmd.exe) (C:\Windows\System32\cmd.exe c:\windows\system32\cmd.exe) (C:\Windows\System32\\net.exe net  user)",
    "(C: Windows System32 net.exe net user administrator /domain')",
    "(C: Windows System32 Windows PowerShell v1.<PID> powershell.exe powershell -command Get-Service seg120 | select  -Expand Display Name |out-file  -append test.txt ')",
    "(C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command Get-Process | tasklist.exe)",
    "(C:/Windows/System32/cmd.exe) <STEP> (C:/Windows/System32/netsh.exe netsh interface portproxy)"
]

test_labels = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 第一個樣本的正確標籤 (多標籤)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 第二個樣本
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 第三個樣本
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

test_data = {
    "text": test_texts,
    "labels": test_labels
}

# 建立一個測試資料集（使用 Hugging Face 的 Dataset）
from datasets import Dataset
#test_data = {"text": test_texts}
test_dataset = Dataset.from_dict(test_data)

# 對測試資料使用與訓練資料相同的 tokenization 函數進行處理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)
print(test_dataset[0])
#test_dataset = test_dataset.map(lambda e: {"labels": [float(v) for v in e["labels"]]}, batched=False)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
#print(test_dataset[0])
print(f"Test dataset size: {len(test_dataset)}")

# 使用 trainer.predict() 得到模型預測
predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions  # 這部分是 raw logits

# 針對多標籤分類，進行 sigmoid 並設定 threshold（例如 0.5）判斷是否激活該標籤
pred_tensor = torch.sigmoid(torch.tensor(logits))
pred_multi_hot = (pred_tensor > 0.5).int()

# 假設你有一個 id2label 字典（例如用來將數值映射回標籤描述），這裡僅示範回傳數值
print("\n====== 測試結果 ======")
for i, text in enumerate(test_texts):
    print(f"\n第 {i+1} 筆輸入文本: {text}")
    logits_per_sample = logits[i]
    sigmoid_probs = torch.sigmoid(torch.tensor(logits_per_sample))
    multi_hot_pred = (sigmoid_probs > 0.5).int()

    print("Raw logits:", logits_per_sample)
    print("Sigmoid 機率:", sigmoid_probs.tolist())
    print("Multi-hot 預測:", multi_hot_pred.tolist())

    #將 multi-hot 向量中的 index 取出
    print(pred_multi_hot)
    active_indices = [j for j, val in enumerate(pred_multi_hot[i]) if val == 1]
    print("輸入文本:", text)
    #print('\033[0m', end='')    # <- reset all styles
    print("預測標籤 (索引形式):", active_indices)
    # 如果你有 id2label，可以做進一步映射：
    pred_labels = [id2label[j] for j in active_indices]
    print("預測標籤:", pred_labels)
    print("-" * 50)

# 第二筆測資專屬檢查
logits_2nd = logits[1]
sigmoid_2nd = torch.sigmoid(torch.tensor(logits_2nd))
multi_hot_2nd = (sigmoid_2nd > 0.5).int()
active_indices_2nd = [j for j, val in enumerate(multi_hot_2nd) if val == 1]

print("\n====== 第二筆測試資料 ======")
print("Raw logits:", logits_2nd)
print("Sigmoid 機率:", sigmoid_2nd.tolist())
print("Multi-hot 預測:", multi_hot_2nd.tolist())
print("預測標籤 (索引形式):", active_indices_2nd)
pred_labels_2nd = [id2label[j] for j in active_indices_2nd]
print("預測標籤:", pred_labels_2nd)
print("-" * 50)

import json
output_dir = "checkpoints_Six/full_model"
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
with open(f"{output_dir}/id2label.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)
print(f"模型、tokenizer 和标签映射已保存到 {output_dir}")

model = AutoModelForSequenceClassification.from_pretrained("checkpoints_Six/full_model")
device = torch.device("cpu")
model.to(device)
model.eval()
#preds_output = trainer.predict(test_dataset)
#logits = preds_output.predictions  # logits 形状就是 (N, num_labels)，其中 N= len(test_dataset)

# 2. 确保不记录梯度
with torch.no_grad():
    single = test_dataset[0]                     # HuggingFace Dataset 的第 0 条
    ids  = single["input_ids"].unsqueeze(0).to(device)
    mask = single["attention_mask"].unsqueeze(0).to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits  # shape=[1, num_labels]

    probs = torch.sigmoid(logits).cpu().tolist()[0]
    print("Sigmoid 概率：", probs)

# 假設 trainer.train() 執行完之後 trainer.state.log_history 內就有訓練日誌
log_history = trainer.state.log_history

# 儲存到 log_history.json
with open("log_historynew_Six.json", "w") as f:
    json.dump(log_history, f, indent=2)

print("Log history 已儲存到 log_history.json")

