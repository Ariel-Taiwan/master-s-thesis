# master's-thesis
Multi-Label TTP Annotation for Attacker Command Sequences in AD Honeypots using BERT
基於 BERT 的方法，用於在 Active Directory 蜂巢環境中對攻擊者的指令序列進行多標籤的 TTP標註。藉由分析 Windows EVTX 日誌所還原的進程鏈，系統能夠自動識別出對應的 MITRE ATT&CK 技術，並進一步去評估每個 TTP 的風險嚴重性，協助資安人員了解攻擊與辨別威脅的嚴重程度。

reconsturct_attack_chain_all_folder's_evtx_v7.py: 重建進程鏈.py

preprocessing_addtoken_Focalloss.py: 訓練程式碼，微調版的CySecBert.py

attack_TS_summary_v6.py: 計算攻擊TTP嚴重度分數.py

analysis_all_in_one_4_graphs.py: 畫出訓練成果圖，如下：

<img width="970" height="574" alt="Six" src="https://github.com/user-attachments/assets/2f766e88-81a2-44cd-b449-38ff6d00e353" />


<img width="774" height="574" alt="Six_AUC" src="https://github.com/user-attachments/assets/221917ab-e47f-411e-9457-5d69fcb38d73" />
