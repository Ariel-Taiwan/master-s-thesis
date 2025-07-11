import os
from Evtx.Evtx import Evtx
from xml.etree import ElementTree as ET
from datetime import datetime

def parse_evtx(file_path, process_data):
    """
    解析 EVTX 檔案中的事件，提取 NewProcessId、CommandLine、
    ParentProcessId、ParentProcessName 以及 SystemTime。
    如果該事件有 ParentProcessId，則直接從 process_data 中找出時間上最接近的父事件，
    並將該父事件的祖先鏈條附加到當前事件的 Ancestry 欄位中。
    """
    namespace = {'ns': 'http://schemas.microsoft.com/win/2004/08/events/event'}
    with Evtx(file_path) as log:
        for record in log.records():
            try:
                xml = ET.fromstring(record.xml())
                event_id_elem = xml.find(".//ns:EventID", namespace)
                if event_id_elem is None:
                    continue

                new_proc_elem = xml.find(".//ns:Data[@Name='NewProcessId']", namespace)
                if new_proc_elem is None:
                    continue
                new_process_id = new_proc_elem.text

                cmd_elem = xml.find(".//ns:Data[@Name='CommandLine']", namespace)
                command_line = cmd_elem.text if cmd_elem is not None else ""

                new_process_name_elem = xml.find(".//ns:Data[@Name='NewProcessName']", namespace)
                new_process_name = new_process_name_elem.text if new_process_name_elem is not None else ""

                parent_proc_elem = xml.find(".//ns:Data[@Name='ProcessId']", namespace)
                parent_process_id = parent_proc_elem.text if parent_proc_elem is not None else ""

                parent_cmd_elem = xml.find(".//ns:Data[@Name='ParentProcessName']", namespace)
                parent_command_line = parent_cmd_elem.text if parent_cmd_elem is not None else ""

                time_elem = xml.find(".//ns:TimeCreated", namespace)
                system_time = None
                if time_elem is not None and "SystemTime" in time_elem.attrib:
                    st = time_elem.attrib["SystemTime"].rstrip("Z")
                    if "." in st:
                        main_part, fraction = st.split(".", 1)
                        fraction = fraction[:6]
                        st = main_part + "." + fraction
                    system_time = datetime.fromisoformat(st)

                if not command_line and not new_process_name:
                    continue

                event_record = {
                    "ProcessId": new_process_id,
                    "CommandLine": command_line,
                    "NewProcessName": new_process_name,
                    "ParentProcessId": parent_process_id,
                    "ParentCommandLine": parent_command_line,
                    "SystemTime": system_time,
                    "Ancestry": []
                }

                if parent_process_id and parent_process_id in process_data:
                    candidate = None
                    for pevent in process_data[parent_process_id]:
                        if pevent["SystemTime"] <= system_time and pevent["NewProcessName"] == parent_command_line:
                            if candidate is None or pevent["SystemTime"] > candidate["SystemTime"]:
                                candidate = pevent
                    if candidate:
                        event_record["Ancestry"] = candidate["Ancestry"] + [candidate]

                process_data.setdefault(new_process_id, []).append(event_record)

            except Exception:
                continue


def build_process_chains(process_data):
    """
    根據每個事件的 Ancestry 產生父子鏈條，返回每條鏈條為事件列表
    """
    chains = []
    for event_list in process_data.values():
        for event in event_list:
            chain = event["Ancestry"] + [event]
            chains.append(chain)
    return chains


def event_str(e):
    desc = f"{e['NewProcessName']} {e['CommandLine']}'.strip()"
    return f"{e['ProcessId']} ({desc})"


def format_chain(chain):
    formatted = []
    first = chain[0]
    if first["ParentProcessId"] and first["ParentCommandLine"]:
        formatted.append(f"{first['ParentProcessId']} ({first['ParentCommandLine']})")
    for event in chain:
        formatted.append(event_str(event))
    deduped = []
    for item in formatted:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return " -> ".join(deduped)


def main():
    folder_path = "/home/ariel612410026/python/myapp/EVTX-ATTACK-SAMPLES"  # 替換為你的資料夾路徑
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # 遞迴尋找所有 .evtx 檔案
    evtx_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.evtx'):
                evtx_files.append(os.path.join(root, file))

    if not evtx_files:
        print("No .evtx files found in the folder or its subfolders.")
        return

    process_data = {}
    for evtx_file in evtx_files:
        print(f"Parsing: {evtx_file}")
        parse_evtx(evtx_file, process_data)
        print(f"Done File: {evtx_file}")

    chains = build_process_chains(process_data)

    # 過濾完整唯一鏈條
    max_chains = {}
    for chain in chains:
        leaf_id = chain[-1]["ProcessId"]
        if leaf_id not in max_chains or len(chain) > len(max_chains[leaf_id]):
            max_chains[leaf_id] = chain
    unique_chains = list(max_chains.values())

    output_file = "EVTX-to-MITRE-Attack/process_chains_output_test_for_which.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        #f.write("進程鏈條:\n\n")
        if unique_chains:
            for chain in unique_chains:
                f.write(format_chain(chain) + "\n")
        else:
            f.write("未找到任何進程鏈條。\n")

    print(f"結果已輸出至 {output_file}")

if __name__ == "__main__":
    main()

