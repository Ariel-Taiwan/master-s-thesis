import json
import matplotlib.pyplot as plt
import numpy as np

# 讀取 log_history
with open("log_historynew_Six.json", "r") as f:
    log_history = json.load(f)

# 分類訓練與評估紀錄
train_logs = [log for log in log_history if "loss" in log]
eval_logs = [log for log in log_history if "eval_loss" in log]

from collections import defaultdict
epoch_best_eval = {}
epoch_best_loss = defaultdict(lambda: float("inf"))

for log in eval_logs:
    epoch = log["epoch"]
    loss = log["eval_loss"]
    if loss < epoch_best_loss[epoch]:
        epoch_best_eval[epoch] = log
        epoch_best_loss[epoch] = loss

eval_logs = list(epoch_best_eval.values())

# 訓練 Loss
train_epochs = [log["epoch"] for log in train_logs]
train_losses = [log["loss"] for log in train_logs]

# 評估 Loss
eval_epochs = [log["epoch"] for log in eval_logs]
eval_losses = [log["eval_loss"] for log in eval_logs]

# Micro 指標
precision_micro = [log["eval_precision_micro"] for log in eval_logs]
recall_micro = [log["eval_recall_micro"] for log in eval_logs]
f1_micro = [log["eval_f1_micro"] for log in eval_logs]

# Macro 指標
precision_macro = [log["eval_precision_macro"] for log in eval_logs]
recall_macro = [log["eval_recall_macro"] for log in eval_logs]
f1_macro = [log["eval_f1_macro"] for log in eval_logs]

# AUC macro
roc_auc_macro = [log["eval_roc_auc_macro"] for log in eval_logs]

# 處理 NaN → np.nan
roc_auc_macro = [val if val == val else np.nan for val in roc_auc_macro]

max_epoch = max(train_epochs)
mask = [e <= max_epoch for e in eval_epochs]

eval_epochs     = [e   for e, m in zip(eval_epochs,     mask) if m]
eval_losses     = [l   for l, m in zip(eval_losses,     mask) if m]
precision_micro = [p   for p, m in zip(precision_micro, mask) if m]
recall_micro    = [r   for r, m in zip(recall_micro,    mask) if m]
f1_micro        = [f1  for f1,m in zip(f1_micro,        mask) if m]
precision_macro = [p   for p, m in zip(precision_macro, mask) if m]
recall_macro    = [r   for r, m in zip(recall_macro,    mask) if m]
f1_macro        = [f1  for f1,m in zip(f1_macro,        mask) if m]
roc_auc_macro   = [a   for a, m in zip(roc_auc_macro,   mask) if m]

# 畫圖
plt.figure(figsize=(10, 6))

# Loss 曲線
plt.subplot(2, 2, 1)
plt.plot(train_epochs, train_losses, label="Training Loss")
plt.plot(eval_epochs, eval_losses, label="Evaluation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss")
plt.legend()
plt.grid(True)

# Micro Precision/Recall/F1
plt.subplot(2, 2, 2)
plt.plot(eval_epochs, precision_micro, label="Precision (micro)")
plt.plot(eval_epochs, recall_micro, label="Recall (micro)")
plt.plot(eval_epochs, f1_micro, label="F1-score (micro)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Micro Precision, Recall, F1-score")
plt.legend()
plt.grid(True)

# Macro Precision/Recall/F1
plt.subplot(2, 2, 3)
plt.plot(eval_epochs, precision_macro, label="Precision (macro)")
plt.plot(eval_epochs, recall_macro, label="Recall (macro)")
plt.plot(eval_epochs, f1_macro, label="F1-score (macro)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Macro Precision, Recall, F1-score")
plt.legend()
plt.grid(True)

# AUC macro
plt.subplot(2, 2, 4)
plt.plot(eval_epochs, roc_auc_macro, label="ROC AUC (macro)", color="purple")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("ROC AUC (macro)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

