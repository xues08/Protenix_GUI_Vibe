import sys
import os
import json
import subprocess
import re
import glob
import platform
from datetime import datetime
import shlex
import csv
import shutil
import random
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QStackedWidget, 
                             QFrame, QLineEdit, QComboBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QScrollArea, QGridLayout, QTextEdit, QSlider, QAbstractItemView, QAbstractScrollArea,
                             QMessageBox, QFileDialog, QDialog, QTabWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QFont, QColor, QCursor, QTextCursor, QPainter, QPen, QKeySequence
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None
    WEB_ENGINE_AVAILABLE = False


try:
    from send2trash import send2trash
except ImportError:
    send2trash = None

def safe_trash_delete(path):
    if not os.path.exists(path):
        return
    if send2trash:
        send2trash(path)
    else:
        import shutil
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


# --- 样式表 (QSS) 模拟现代 Web UI ---
STYLE_SHEET = """
QMainWindow {
    background-color: #f8fafc;
}
/* 侧边栏 */
#Sidebar {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
#SidebarButton {
    text-align: left;
    padding: 10px 15px;
    border: none;
    border-radius: 6px;
    background-color: transparent;
    color: #475569;
    font-size: 13px;
}
#SidebarButton:hover {
    background-color: #f1f5f9;
    color: #0f172a;
}
#SidebarButton:checked {
    background-color: #eff6ff;
    color: #2563eb;
    font-weight: bold;
    border: 1px solid #bfdbfe;
}
/* 卡片 (Cards) */
#Card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
#CardTitle {
    font-size: 16px;
    font-weight: bold;
    color: #1e293b;
    margin-bottom: 8px;
}
/* 输入框与下拉菜单 */
QLineEdit, QComboBox, QTextEdit {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px 10px;
    background-color: #f8fafc;
    color: #334155;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #3b82f6;
    background-color: #ffffff;
}
/* 日志控制台 */
#LogConsole {
    background-color: #0f172a;
    color: #10b981;
    font-family: Consolas, monospace;
    font-size: 12px;
}
/* 按钮 */
QPushButton {
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 13px;
}
#PrimaryButton {
    background-color: #2563eb;
    color: white;
    border: none;
    font-weight: bold;
    padding: 8px 20px;
    border-radius: 20px;
}
#PrimaryButton:hover { background-color: #1d4ed8; }
#PrimaryButton:disabled { background-color: #94a3b8; }
#OutlineButton {
    background-color: transparent;
    color: #2563eb;
    border: 1px solid #93c5fd;
    border-radius: 15px;
}
#OutlineButton:hover { background-color: #eff6ff; }
/* 切换按钮组 (Toggle Group) */
#ToggleLeft {
    border: 1px solid #cbd5e1;
    border-top-left-radius: 4px;
    border-bottom-left-radius: 4px;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    background-color: #ffffff;
    border-right: none;
}
#ToggleRight {
    border: 1px solid #cbd5e1;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    background-color: #ffffff;
}
#ToggleLeft:checked, #ToggleRight:checked {
    background-color: #eff6ff;
    color: #2563eb;
    border-color: #3b82f6;
}
/* 表格 */
QTableWidget {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
    gridline-color: #f1f5f9;
}
QHeaderView::section {
    background-color: #f8fafc;
    border: none;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px;
    font-weight: bold;
    color: #475569;
}
"""



# --- 核心预测执行线程 (QThread) ---
class ProtenixWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, job_data, out_dir, script_path, cuda_home, device, model_name, use_msa, use_template, use_rna_msa, seeds, sample_num, recycle, diffusion_steps, existing_json_path=None):
        super().__init__()
        self.job_data = job_data
        self.out_dir = out_dir
        self.script_path = script_path
        self.cuda_home = cuda_home
        self.device = device
        self.model_name = model_name
        self.use_msa = use_msa
        self.use_template = use_template
        self.use_rna_msa = use_rna_msa
        self.seeds = seeds
        self.sample_num = sample_num
        self.recycle = recycle
        self.diffusion_steps = diffusion_steps
        self.existing_json_path = existing_json_path

    def run(self):
        try:
            # 1. 确保输出目录存在
            os.makedirs(self.out_dir, exist_ok=True)
            
            # 2. 生成或使用已有的 input.json
            if self.existing_json_path and os.path.exists(self.existing_json_path):
                input_json_path = self.existing_json_path
                self.log_signal.emit(f"[*] Using existing JSON file: {input_json_path}")
            else:
                input_json_path = os.path.join(self.out_dir, "input.json")
                with open(input_json_path, 'w', encoding='utf-8') as f:
                    if isinstance(self.job_data, list):
                        json.dump(self.job_data, f, indent=2)
                    else:
                        json.dump([self.job_data], f, indent=2)
                self.log_signal.emit(f"[*] Generated new Input JSON at: {input_json_path}")
            
            # 3. 智能构造 Protenix 命令行指令
            script_parts = self.script_path.strip().split()
            
            if script_parts[0].endswith('.py'):
                if not os.path.exists(script_parts[0]):
                    self.finished_signal.emit(False, f"Script not found at: {script_parts[0]}\nPlease use the 'Browse' button to select the correct Python script.")
                    return
                cmd = [sys.executable] + script_parts + [
                    "--input_json_path", input_json_path,
                    "--dump_dir", self.out_dir,
                    "--model_name", self.model_name
                ]
                if self.use_msa:
                    cmd.extend(["--use_msa", "True"])
                if self.use_template:
                    cmd.extend(["--use_template", "True"])
                if self.use_rna_msa:
                    cmd.extend(["--use_rna_msa", "True"])
                if self.seeds:
                    cmd.extend(["--seeds", str(self.seeds)])
                
                cmd.extend([
                    "--cycle", str(self.recycle),
                    "--step", str(self.diffusion_steps),
                    "--sample", str(self.sample_num)
                ])
            else:
                cmd = script_parts
                if cmd[0] == "protenix" and len(cmd) == 1:
                    cmd.append("pred")
                
                cmd.extend([
                    "--input", input_json_path,
                    "--out_dir", self.out_dir,
                    "--model_name", self.model_name
                ])
                if self.use_msa:
                    cmd.extend(["--use_msa", "True"])
                if self.use_template:
                    cmd.extend(["--use_template", "True"])
                if self.use_rna_msa:
                    cmd.extend(["--use_rna_msa", "True"])
                if self.seeds:
                    cmd.extend(["--seeds", str(self.seeds)])
                    
                cmd.extend([
                    "--cycle", str(self.recycle),
                    "--step", str(self.diffusion_steps),
                    "--sample", str(self.sample_num)
                ])
            
            # 4. 配置运行环境与设备参数
            run_env = os.environ.copy()
            if self.device == "CPU":
                self.log_signal.emit("[*] Running in Pure CPU mode (ignoring CUDA)")
                run_env["CUDA_VISIBLE_DEVICES"] = ""  
                cmd.extend([
                    "--triatt_kernel", "torch",
                    "--trimul_kernel", "torch"
                ])
            else:
                if self.cuda_home:
                    run_env["CUDA_HOME"] = self.cuda_home
                    self.log_signal.emit(f"[*] Setting CUDA_HOME to: {self.cuda_home}")

            # 格式化输出命令行
            formatted_cmd = []
            current_line = f"{cmd[0]}"
            for arg in cmd[1:]:
                if arg.startswith("-") or arg.startswith("runner."):
                    formatted_cmd.append(current_line + " \\")
                    current_line = f"    {arg}"
                else:
                    current_line += f' "{arg}"' if " " in arg else f" {arg}"
            formatted_cmd.append(current_line)
            
            self.log_signal.emit("[*] Executing command:")
            for line in formatted_cmd:
                self.log_signal.emit(line)
            self.log_signal.emit("-" * 40)

            # 5. 执行命令并实时获取输出
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                universal_newlines=True,
                env=run_env
            )
            
            for line in self.process.stdout:
                if self.isInterruptionRequested():
                    break
                self.log_signal.emit(line.strip())
            
            self.process.wait()
            
            # 6. 返回执行结果
            if self.isInterruptionRequested() or self.process.returncode == -9 or self.process.returncode == 137: # SIGKILL
                self.finished_signal.emit(False, f"Process aborted by user.")
            elif self.process.returncode == 0:
                self.finished_signal.emit(True, f"Prediction completed successfully. Results in: {self.out_dir}")
            else:
                self.finished_signal.emit(False, f"Process exited with error code {self.process.returncode}")
                
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.requestInterruption()
        if hasattr(self, 'process') and self.process:
            try:
                self.process.kill()
            except Exception:
                pass


class MSAMappingDialog(QDialog):
    def __init__(self, sequences, msa_files, parent=None):
        super().__init__(parent)
        self.sequences = sequences  # List of dicts with info about sequences requiring MSA
        self.msa_files = [""] + msa_files # Add empty option
        self.mapping_result = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Manual MSA Mapping")
        self.resize(800, 500)
        layout = QVBoxLayout(self)

        info_label = QLabel("Automatic MSA matching failed or was ambiguous. Please manually map the files.")
        info_label.setStyleSheet("color: #b91c1c; font-weight: bold;")
        layout.addWidget(info_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Sequence Type/Idx", "Paired MSA", "Unpaired MSA", "Templates"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(len(self.sequences))

        self.combos = []

        for row, seq_info in enumerate(self.sequences):
            seq_type = seq_info['type']
            idx = seq_info['idx']
            
            # Display type and index
            item = QTableWidgetItem(f"{seq_type} (Idx: {idx})")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)

            row_combos = {}
            
            if seq_type == "Protein":
                # Paired
                cb_paired = QComboBox()
                cb_paired.addItems(self.msa_files)
                self.table.setCellWidget(row, 1, cb_paired)
                row_combos['paired'] = cb_paired

                # Unpaired
                cb_unpaired = QComboBox()
                cb_unpaired.addItems(self.msa_files)
                self.table.setCellWidget(row, 2, cb_unpaired)
                row_combos['unpaired'] = cb_unpaired

                # Templates
                cb_templates = QComboBox()
                cb_templates.addItems(self.msa_files)
                self.table.setCellWidget(row, 3, cb_templates)
                row_combos['templates'] = cb_templates
            else:
                # RNA only needs unpaired usually, but we allow selection in unpaired column
                item_na = QTableWidgetItem("N/A")
                item_na.setFlags(item_na.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 1, item_na)
                
                cb_unpaired = QComboBox()
                cb_unpaired.addItems(self.msa_files)
                self.table.setCellWidget(row, 2, cb_unpaired)
                row_combos['unpaired'] = cb_unpaired
                
                item_na2 = QTableWidgetItem("N/A")
                item_na2.setFlags(item_na2.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 3, item_na2)

            self.combos.append(row_combos)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("Confirm Mapping")
        ok_btn.setObjectName("PrimaryButton")
        ok_btn.clicked.connect(self.accept_mapping)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def accept_mapping(self):
        self.mapping_result = []
        for i, seq_info in enumerate(self.sequences):
            row_combos = self.combos[i]
            mapping = {}
            if seq_info['type'] == "Protein":
                p = row_combos['paired'].currentText()
                u = row_combos['unpaired'].currentText()
                t = row_combos['templates'].currentText()
                if p: mapping['pairedMsaPath'] = p
                if u: mapping['unpairedMsaPath'] = u
                if t: mapping['templatesPath'] = t
            else:
                u = row_combos['unpaired'].currentText()
                if u: mapping['unpairedMsaPath'] = u
                
            self.mapping_result.append({
                'seq_idx_global': seq_info['global_idx'],
                'mapping': mapping
            })
            
        self.accept()


# --- 自定义 UI 组件 ---
class Card(QFrame):
    def __init__(self, title=""):
        super().__init__()
        self.setObjectName("Card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("CardTitle")
            self.layout.addWidget(title_label)
            
    def add_widget(self, widget):
        self.layout.addWidget(widget)
        
    def add_layout(self, layout):
        self.layout.addLayout(layout)



class ModificationWidget(QFrame):
    def __init__(self, mod_number, parent=None, is_protein=True):
        super().__init__(parent)
        self.mod_number = mod_number
        self.is_protein = is_protein
        self.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 4px;
                padding: 6px;
                margin: 3px 0;
            }
        """)
        self.create_ui()
        
    def create_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        title_label = QLabel(f"Modification {self.mod_number}:")
        title_label.setStyleSheet("font-weight: bold; color: #475569; font-size: 11px;")
        layout.addWidget(title_label)
        
        if self.is_protein:
            layout.addWidget(QLabel("<span style='color:red;'>*</span> PTM pos"))
            self.ptm_position = QLineEdit()
            self.ptm_position.setPlaceholderText("1")
            self.ptm_position.setFixedWidth(50)
            layout.addWidget(self.ptm_position)
            
            layout.addWidget(QLabel("<span style='color:red;'>*</span> PTM type"))
            self.ptm_type = QLineEdit()
            self.ptm_type.setPlaceholderText("CCD_")
            self.ptm_type.setFixedWidth(80)
            layout.addWidget(self.ptm_type)
        else:
            layout.addWidget(QLabel("<span style='color:red;'>*</span> Base pos"))
            self.ptm_position = QLineEdit()
            self.ptm_position.setPlaceholderText("1")
            self.ptm_position.setFixedWidth(50)
            layout.addWidget(self.ptm_position)
            
            layout.addWidget(QLabel("<span style='color:red;'>*</span> Mod type"))
            self.ptm_type = QLineEdit()
            self.ptm_type.setPlaceholderText("CCD_")
            self.ptm_type.setFixedWidth(80)
            layout.addWidget(self.ptm_type)
        
        layout.addStretch()
        
        delete_btn = TrashButton()
        delete_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        delete_btn.clicked.connect(self.on_delete_clicked)
        layout.addWidget(delete_btn)

def normalize_sequence_text(text, is_ligand_or_ion=False):
    if is_ligand_or_ion:
        # For ligands (like SMILES or FILE_ paths) or ions, 
        # we only strip leading/trailing whitespace and DO NOT convert to uppercase
        return text.strip()
        
    lines = [line for line in text.splitlines() if not line.strip().startswith(">")]
    merged = "".join(lines)
    merged = re.sub(r"\s+", "", merged)
    return merged.upper()

def extract_plddt_from_cif(cif_content):
    lines = cif_content.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            i += 1
            cols = []
            while i < len(lines) and lines[i].strip().startswith("_atom_site."):
                cols.append(lines[i].strip())
                i += 1
            if cols:
                try:
                    idx = cols.index("_atom_site.B_iso_or_equiv")
                except ValueError:
                    idx = -1
                if idx >= 0:
                    values = []
                    while i < len(lines):
                        line = lines[i].strip()
                        if not line or line.startswith("#") or line.startswith("loop_") or line.startswith("_"):
                            break
                        parts = shlex.split(line)
                        if len(parts) >= len(cols):
                            try:
                                values.append(float(parts[idx]))
                            except Exception:
                                pass
                        i += 1
                    if values:
                        return values
            continue
        i += 1
    return []

def validate_sequence_by_type(sequence, seq_type, label):
    if seq_type == "proteinChain":
        allowed = set("ACDEFGHIKLMNPQRSTVWYX")
        if any(ch not in allowed for ch in sequence):
            return f"{label}: 'proteinChain' contains invalid characters (e.g., numbers or symbols)."
    elif seq_type == "dnaSequence":
        allowed = set("ATGCN")
        if any(ch not in allowed for ch in sequence):
            return f"{label}: 'dnaSequence' contains invalid characters."
    elif seq_type == "rnaSequence":
        allowed = set("AUGCN")
        if any(ch not in allowed for ch in sequence):
            return f"{label}: 'rnaSequence' contains invalid characters."
    return None
        
    def on_delete_clicked(self):
        # 查找正确的父对象
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_modification'):
            parent = parent.parent()
        if parent and hasattr(parent, 'remove_modification'):
            parent.remove_modification(self)
        
    def get_data(self):
        if self.is_protein:
            return {
                "ptmType": self.ptm_type.text().strip(),
                "ptmPosition": int(self.ptm_position.text().strip())
            }
        else:
            return {
                "modificationType": self.ptm_type.text().strip(),
                "basePosition": int(self.ptm_position.text().strip())
            }
        
    def is_valid(self):
        return bool(self.ptm_type.text().strip() and self.ptm_position.text().strip())
        
    def set_is_protein(self, is_protein):
        # 简单更新标签而不是重建整个UI
        if self.is_protein == is_protein:
            return
            
        self.is_protein = is_protein
        
        # 找到标签widget并更新文本
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, QLabel):
                text = widget.text()
                if "PTM pos" in text:
                    widget.setText("<span style='color:red;'>*</span> Base pos")
                elif "Base pos" in text:
                    widget.setText("<span style='color:red;'>*</span> PTM pos")
                elif "PTM type" in text:
                    widget.setText("<span style='color:red;'>*</span> Mod type")
                elif "Mod type" in text:
                    widget.setText("<span style='color:red;'>*</span> PTM type")

class SequenceWidget(QFrame):
    def __init__(self, seq_number, parent=None):
        super().__init__(parent)
        self.seq_number = seq_number
        self.modifications = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
                margin: 6px 0;
            }
        """)
        self.create_ui()
        
    def create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        title_layout = QHBoxLayout()
        
        title_label = QLabel(f"Sequence {self.seq_number}")
        title_label.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 12px;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        if self.seq_number > 1:
            delete_btn = TrashButton()
            delete_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: transparent;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                }
            """)
            delete_btn.clicked.connect(self.on_delete_clicked)
            title_layout.addWidget(delete_btn)
        
        layout.addLayout(title_layout)
        
        meta_layout = QHBoxLayout()
        
        meta_layout.addWidget(QLabel("<span style='color:red;'>*</span> Molecule type"))
        self.mol_combo = QComboBox()
        self.mol_combo.addItems(["Protein", "RNA", "DNA", "Ligand", "Ion"])
        self.mol_combo.setMinimumWidth(90)
        self.mol_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mol_combo.currentTextChanged.connect(self.on_mol_type_changed)
        meta_layout.addWidget(self.mol_combo)
        
        meta_layout.addSpacing(10)
        
        meta_layout.addWidget(QLabel("<span style='color:red;'>*</span> Copy"))
        self.inp_copy = QLineEdit("1")
        self.inp_copy.setMinimumWidth(28)
        self.inp_copy.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        meta_layout.addWidget(self.inp_copy)
        
        meta_layout.addSpacing(10)
        
        meta_layout.addWidget(QLabel("ID (optional)"))
        self.inp_id = QLineEdit()
        self.inp_id.setPlaceholderText("e.g., H,L")
        self.inp_id.setMinimumWidth(48)
        self.inp_id.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        meta_layout.addWidget(self.inp_id)
        
        meta_layout.addStretch()
        layout.addLayout(meta_layout)
        
        # Sequence
        seq_layout = QHBoxLayout()
        self.seq_label = QLabel("<span style='color:red;'>*</span> Sequence")
        seq_layout.addWidget(self.seq_label)
        
        seq_layout.addStretch()
        
        self.btn_browse_ligand = QPushButton("Browse File...")
        self.btn_browse_ligand.setObjectName("OutlineButton")
        self.btn_browse_ligand.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self.btn_browse_ligand.setVisible(False)
        self.btn_browse_ligand.clicked.connect(self.browse_ligand_file)
        seq_layout.addWidget(self.btn_browse_ligand)
        
        layout.addLayout(seq_layout)
        
        self.seq_text = QTextEdit()
        self.seq_text.setPlaceholderText("Enter amino acid or nucleotide sequence here...")
        self.seq_text.setFixedHeight(80)
        self.seq_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.seq_text)
        
        self.ligand_hint = QLabel('<i>Find SMILES at <a href="https://www.rcsb.org/search/chemical">RCSB</a>. For CCD, use prefix "CCD_" (e.g., "CCD_ATP" or "CCD_NAG_BMA_BGC" for glycans). For structure files, use prefix "FILE_" followed by path (PDB/SDF/MOL/MOL2).</i>')
        self.ligand_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        self.ligand_hint.setOpenExternalLinks(True)
        self.ligand_hint.setWordWrap(True)
        self.ligand_hint.setVisible(False)
        layout.addWidget(self.ligand_hint)
        
        self.ion_hint = QLabel('<i>For Ion CCD codes, DO NOT include "CCD_". For example, use "NA" for Sodium, "MG" for Magnesium.</i>')
        self.ion_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        self.ion_hint.setWordWrap(True)
        self.ion_hint.setVisible(False)
        layout.addWidget(self.ion_hint)
        
        msa_header = QHBoxLayout()
        self.msa_toggle = ArrowButton()
        self.msa_toggle.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        self.msa_toggle.set_expanded(False)
        self.msa_toggle.clicked.connect(self.toggle_msa_fields)
        msa_header.addWidget(self.msa_toggle)
        msa_title = QLabel("MSA / Templates")
        msa_title.setStyleSheet("color: #475569; font-size: 12px; font-weight: 600;")
        msa_header.addWidget(msa_title)
        msa_header.addStretch()
        layout.addLayout(msa_header)
        
        # Protein-specific fields (MSA and templates)
        self.protein_fields = QWidget()
        protein_layout = QGridLayout(self.protein_fields)
        protein_layout.setContentsMargins(0, 0, 0, 0)
        protein_layout.setSpacing(6)
        
        protein_layout.addWidget(QLabel("Paired MSA Path"), 0, 0)
        self.paired_msa_path = QLineEdit()
        self.paired_msa_path.setPlaceholderText("/path/to/pairing.a3m")
        self.paired_msa_path.setMinimumWidth(140)
        self.paired_msa_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        protein_layout.addWidget(self.paired_msa_path, 0, 1)
        
        browse_paired_btn = QPushButton("Browse")
        browse_paired_btn.setFixedSize(70, 24)
        browse_paired_btn.clicked.connect(lambda: self.browse_file(self.paired_msa_path, "MSA Files (*.a3m)"))
        protein_layout.addWidget(browse_paired_btn, 0, 2)
        
        protein_layout.addWidget(QLabel("Unpaired MSA Path"), 1, 0)
        self.unpaired_msa_path = QLineEdit()
        self.unpaired_msa_path.setPlaceholderText("/path/to/non_pairing.a3m")
        self.unpaired_msa_path.setMinimumWidth(140)
        self.unpaired_msa_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        protein_layout.addWidget(self.unpaired_msa_path, 1, 1)
        
        browse_unpaired_btn = QPushButton("Browse")
        browse_unpaired_btn.setFixedSize(70, 24)
        browse_unpaired_btn.clicked.connect(lambda: self.browse_file(self.unpaired_msa_path, "MSA Files (*.a3m)"))
        protein_layout.addWidget(browse_unpaired_btn, 1, 2)
        
        protein_layout.addWidget(QLabel("Templates Path"), 2, 0)
        self.templates_path = QLineEdit()
        self.templates_path.setPlaceholderText("/path/to/templates.a3m")
        self.templates_path.setMinimumWidth(140)
        self.templates_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        protein_layout.addWidget(self.templates_path, 2, 1)
        
        browse_templates_btn = QPushButton("Browse")
        browse_templates_btn.setFixedSize(70, 24)
        browse_templates_btn.clicked.connect(lambda: self.browse_file(self.templates_path, "Template Files (*.a3m *.hhr)"))
        protein_layout.addWidget(browse_templates_btn, 2, 2)
        
        self.protein_fields.setVisible(False)
        layout.addWidget(self.protein_fields)
        
        # Modifications section (放在一个容器中方便控制显示)
        self.mods_section = QWidget()
        mods_section_layout = QVBoxLayout(self.mods_section)
        mods_section_layout.setContentsMargins(0, 0, 0, 0)
        mods_section_layout.setSpacing(6)
        
        self.mods_container = QWidget()
        self.mods_layout = QVBoxLayout(self.mods_container)
        self.mods_layout.setContentsMargins(0, 0, 0, 0)
        self.mods_layout.setSpacing(4)
        mods_section_layout.addWidget(self.mods_container)
        
        # Add modification button
        add_mod_btn = QPushButton("+ Add modification")
        add_mod_btn.setObjectName("OutlineButton")
        add_mod_btn.setStyleSheet("font-size: 11px; padding: 4px 12px;")
        add_mod_btn.clicked.connect(self.add_modification)
        mods_section_layout.addWidget(add_mod_btn)
        
        layout.addWidget(self.mods_section)
        
        # 初始设置显示/隐藏
        self.on_mol_type_changed(self.mol_combo.currentText())
        
    def toggle_msa_fields(self):
        expanded = not self.protein_fields.isVisible()
        self.msa_toggle.set_expanded(expanded)
        self.protein_fields.setVisible(expanded)
        
    def browse_ligand_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ligand Structure File",
            "",
            "Structure Files (*.pdb *.sdf *.mol *.mol2 *.PDB *.SDF *.MOL *.MOL2);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if file_path:
            # Backend json_parser.py expects strictly lower case extensions
            path_obj = Path(file_path)
            if path_obj.suffix and any(c.isupper() for c in path_obj.suffix):
                # We create a local copy with a lowercase extension in the same directory
                # so the backend can read it without throwing a ValueError
                lower_ext = path_obj.suffix.lower()
                new_path_str = str(path_obj.with_suffix(lower_ext))
                
                # Only copy if the lowercased filename doesn't already exist
                if not os.path.exists(new_path_str) or os.path.abspath(file_path) == os.path.abspath(new_path_str):
                    try:
                        # Copy the file to the new lowercase extension name
                        shutil.copy2(file_path, new_path_str)
                        file_path = new_path_str
                        QMessageBox.information(self, "Info", f"A copy of the ligand file with a lowercase extension was created for backend compatibility:\n{new_path_str}")
                    except Exception as e:
                        QMessageBox.warning(self, "Warning", f"Could not create a lowercase extension copy. You may need to rename your file manually.\nError: {e}")
                else:
                    # If it exists, just use it
                    file_path = new_path_str

            self.seq_text.setText(f"FILE_{file_path}")
            
    def on_delete_clicked(self):
        # 查找正确的父对象
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_sequence'):
            parent = parent.parent()
        if parent and hasattr(parent, 'remove_sequence'):
            parent.remove_sequence(self)
        
    def add_modification(self):
        mod_number = len(self.modifications) + 1
        is_protein = (self.mol_combo.currentText() == "Protein")
        mod_widget = ModificationWidget(mod_number, self, is_protein)
        self.modifications.append(mod_widget)
        self.mods_layout.addWidget(mod_widget)
        self.renumber_modifications()
        
    def remove_modification(self, mod_widget):
        if mod_widget in self.modifications:
            self.modifications.remove(mod_widget)
            mod_widget.setParent(None)
            self.renumber_modifications()
            
    def renumber_modifications(self):
        for i, mod in enumerate(self.modifications):
            mod.mod_number = i + 1
            mod.findChild(QLabel).setText(f"Modification {i + 1}")
            
    def on_mol_type_changed(self, mol_type):
        # 根据选择改变Sequence标签和提示
        if mol_type == "Ligand":
            self.seq_label.setText("<span style='color:red;'>*</span> SMILES/CCD/FILE_")
            self.ligand_hint.setVisible(True)
            self.ion_hint.setVisible(False)
            self.btn_browse_ligand.setVisible(True)
        elif mol_type == "Ion":
            self.seq_label.setText("<span style='color:red;'>*</span> Sequence")
            self.ligand_hint.setVisible(False)
            self.ion_hint.setVisible(True)
            self.btn_browse_ligand.setVisible(False)
        else:
            self.seq_label.setText("<span style='color:red;'>*</span> Sequence")
            self.ligand_hint.setVisible(False)
            self.ion_hint.setVisible(False)
            self.btn_browse_ligand.setVisible(False)
            
        # 显示/隐藏Protein和RNA特定的字段
        show_msa = (mol_type in ["Protein", "RNA"])
        self.msa_toggle.setVisible(show_msa)
        if not show_msa:
            self.protein_fields.setVisible(False)
            self.msa_toggle.set_expanded(False)
        else:
            if not self.msa_toggle.is_expanded:
                self.protein_fields.setVisible(False)
        
        # 只有Protein、DNA、RNA才显示modifications
        show_mods = (mol_type in ["Protein", "RNA", "DNA"])
        self.mods_section.setVisible(show_mods)
        
        # 更新所有modifications的标签
        if show_mods:
            for mod in self.modifications:
                mod.set_is_protein(is_protein)
            
        # 更新sequence输入框的描述
        if mol_type == "Protein":
            self.seq_text.setPlaceholderText("Enter amino acid sequence here...")
        elif mol_type == "RNA":
            self.seq_text.setPlaceholderText("Enter RNA sequence (A, U, G, C, N) here...")
        elif mol_type == "DNA":
            self.seq_text.setPlaceholderText("Enter DNA sequence (A, T, G, C, N) here...")
        elif mol_type == "Ligand":
            self.seq_text.setPlaceholderText("Enter ligand SMILES string or CCD code here...")
        elif mol_type == "Ion":
            self.seq_text.setPlaceholderText("Enter ion CCD code here...")
            
    def browse_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File", 
            "", 
            file_filter,
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if file_path:
            line_edit.setText(file_path)
            
    def browse_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            "",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if dir_path:
            line_edit.setText(dir_path)
            
    def get_data(self):
        mol_type = self.mol_combo.currentText()
        chain_key = "proteinChain"
        if mol_type == "RNA":
            chain_key = "rnaSequence"
        elif mol_type == "DNA":
            chain_key = "dnaSequence"
        elif mol_type == "Ligand":
            chain_key = "ligand"
        elif mol_type == "Ion":
            chain_key = "ion"
            
        mods_data = []
        for mod in self.modifications:
            if mod.is_valid():
                mods_data.append(mod.get_data())
                
        is_ligand_or_ion = mol_type in ["Ligand", "Ion"]
        cleaned_sequence = normalize_sequence_text(self.seq_text.toPlainText(), is_ligand_or_ion)
        if cleaned_sequence != self.seq_text.toPlainText().strip() and not is_ligand_or_ion:
            # We only force update the UI box if it was a protein/RNA/DNA that got uppercased/cleaned
            self.seq_text.setPlainText(cleaned_sequence)
                
        chain_data = {
            "sequence": cleaned_sequence.strip(),
            "count": int(self.inp_copy.text().strip())
        }
        
        # Add optional id field
        id_text = self.inp_id.text().strip()
        if id_text:
            chain_data["id"] = [x.strip() for x in id_text.split(",")]
            
        # For Ligand, the key is 'ligand' (can be CCD code, file path, or smiles)
        if mol_type == "Ligand":
            chain_data = {
                "ligand": cleaned_sequence.strip(),
                "count": int(self.inp_copy.text().strip())
            }
            if id_text:
                chain_data["id"] = [x.strip() for x in id_text.split(",")]
        
        # For Ion, the key is 'ion' instead of 'name' based on backend json_parser
        elif mol_type == "Ion":
            chain_data = {
                "ion": cleaned_sequence.strip(),
                "count": int(self.inp_copy.text().strip())
            }
            if id_text:
                chain_data["id"] = [x.strip() for x in id_text.split(",")]
        
        # Add protein/RNA-specific fields
        if mol_type in ["Protein", "RNA"]:
            if self.paired_msa_path.text().strip():
                chain_data["pairedMsaPath"] = self.paired_msa_path.text().strip()
            if self.unpaired_msa_path.text().strip():
                chain_data["unpairedMsaPath"] = self.unpaired_msa_path.text().strip()
            if self.templates_path.text().strip():
                chain_data["templatesPath"] = self.templates_path.text().strip()
        
        if mods_data:
            chain_data["modifications"] = mods_data
            
        return {
            chain_key: chain_data
        }
        
    def is_valid(self):
        return bool(self.seq_text.toPlainText().strip() and self.inp_copy.text().strip())

    def validate_sequence(self):
        mol_type = self.mol_combo.currentText()
        is_ligand_or_ion = mol_type in ["Ligand", "Ion"]
        seq = normalize_sequence_text(self.seq_text.toPlainText(), is_ligand_or_ion)
        if seq != self.seq_text.toPlainText().strip() and not is_ligand_or_ion:
            self.seq_text.setPlainText(seq)
        if mol_type == "Protein":
            return validate_sequence_by_type(seq, "proteinChain", f"Sequence {self.seq_number}")
        if mol_type == "DNA":
            return validate_sequence_by_type(seq, "dnaSequence", f"Sequence {self.seq_number}")
        if mol_type == "RNA":
            return validate_sequence_by_type(seq, "rnaSequence", f"Sequence {self.seq_number}")
        return None

class CovalentBondWidget(QFrame):
    def __init__(self, bond_number, parent=None):
        super().__init__(parent)
        self.bond_number = bond_number
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px;
                margin: 6px 0;
            }
        """)
        self.create_ui()
        
    def create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # 标题和删除按钮
        header_layout = QHBoxLayout()
        title_label = QLabel(f"Covalent bond {self.bond_number}")
        title_label.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 12px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        delete_btn = TrashButton()
        delete_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        delete_btn.clicked.connect(self.on_delete_clicked)
        header_layout.addWidget(delete_btn)
        
        layout.addLayout(header_layout)
        
        # Entity 1和Entity 2放在同一行
        entities_layout = QHBoxLayout()
        
        # Entity 1
        grid1 = QGridLayout()
        grid1.setSpacing(4)
        grid1.addWidget(QLabel("<b>Entity 1</b>"), 0, 0, 1, 4)
        grid1.addWidget(QLabel("<span style='color:red;'>*</span> Entity"), 1, 0)
        self.entity1 = QLineEdit()
        self.entity1.setPlaceholderText("0")
        self.entity1.setMinimumWidth(40)
        self.entity1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid1.addWidget(self.entity1, 1, 1)
        
        grid1.addWidget(QLabel("Copy"), 1, 2)
        self.copy1 = QLineEdit()
        self.copy1.setPlaceholderText("0")
        self.copy1.setMinimumWidth(40)
        self.copy1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid1.addWidget(self.copy1, 1, 3)
        
        grid1.addWidget(QLabel("<span style='color:red;'>*</span> Pos"), 2, 0)
        self.position1 = QLineEdit()
        self.position1.setPlaceholderText("1")
        self.position1.setMinimumWidth(40)
        self.position1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid1.addWidget(self.position1, 2, 1)
        
        grid1.addWidget(QLabel("<span style='color:red;'>*</span> Atom"), 2, 2)
        self.atom1 = QLineEdit()
        self.atom1.setPlaceholderText("CA")
        self.atom1.setMinimumWidth(50)
        self.atom1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid1.addWidget(self.atom1, 2, 3)
        
        entities_layout.addLayout(grid1)
        
        # Entity 2
        grid2 = QGridLayout()
        grid2.setSpacing(4)
        grid2.addWidget(QLabel("<b>Entity 2</b>"), 0, 0, 1, 4)
        grid2.addWidget(QLabel("<span style='color:red;'>*</span> Entity"), 1, 0)
        self.entity2 = QLineEdit()
        self.entity2.setPlaceholderText("1")
        self.entity2.setMinimumWidth(40)
        self.entity2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid2.addWidget(self.entity2, 1, 1)
        
        grid2.addWidget(QLabel("Copy"), 1, 2)
        self.copy2 = QLineEdit()
        self.copy2.setPlaceholderText("0")
        self.copy2.setMinimumWidth(40)
        self.copy2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid2.addWidget(self.copy2, 1, 3)
        
        grid2.addWidget(QLabel("<span style='color:red;'>*</span> Pos"), 2, 0)
        self.position2 = QLineEdit()
        self.position2.setPlaceholderText("1")
        self.position2.setMinimumWidth(40)
        self.position2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid2.addWidget(self.position2, 2, 1)
        
        grid2.addWidget(QLabel("<span style='color:red;'>*</span> Atom"), 2, 2)
        self.atom2 = QLineEdit()
        self.atom2.setPlaceholderText("CA")
        self.atom2.setMinimumWidth(50)
        self.atom2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid2.addWidget(self.atom2, 2, 3)
        
        entities_layout.addLayout(grid2)
        layout.addLayout(entities_layout)
        
    def on_delete_clicked(self):
        # 查找正确的父对象
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_bond'):
            parent = parent.parent()
        if parent and hasattr(parent, 'remove_bond'):
            parent.remove_bond(self)
        
    def get_data(self):
        bond_data = {
            "entity1": int(self.entity1.text().strip()),
            "position1": int(self.position1.text().strip()),
            "atom1": self.atom1.text().strip(),
            "entity2": int(self.entity2.text().strip()),
            "position2": int(self.position2.text().strip()),
            "atom2": self.atom2.text().strip()
        }
        
        if self.copy1.text().strip():
            bond_data["copy1"] = int(self.copy1.text().strip())
        if self.copy2.text().strip():
            bond_data["copy2"] = int(self.copy2.text().strip())
            
        return bond_data
        
    def is_valid(self):
        return bool(self.entity1.text().strip() and self.position1.text().strip() and 
                   self.atom1.text().strip() and self.entity2.text().strip() and 
                   self.position2.text().strip() and self.atom2.text().strip())

class ArrowButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_expanded = False
        self.setFixedSize(24, 24)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制箭头
        color = QColor(30, 41, 59)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        
        rect = self.rect()
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        if self.is_expanded:
            # 向下箭头
            points = [
                center_x, center_y - 3,
                center_x + 6, center_y + 3,
                center_x - 6, center_y + 3
            ]
        else:
            # 向右箭头
            points = [
                center_x - 3, center_y - 6,
                center_x + 3, center_y,
                center_x - 3, center_y + 6
            ]
        
        painter.drawPolygon(*[QPoint(points[i], points[i+1]) for i in range(0, len(points), 2)])
        
    def set_expanded(self, expanded):
        self.is_expanded = expanded
        self.update()


class DeleteButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制删除图标（X）
        color = QColor(239, 68, 68)
        painter.setPen(QPen(color, 2.5))
        
        rect = self.rect()
        center_x = rect.center().x()
        center_y = rect.center().y()
        size = 6
        
        # 绘制两条交叉线
        painter.drawLine(center_x - size, center_y - size, center_x + size, center_y + size)
        painter.drawLine(center_x + size, center_y - size, center_x - size, center_y + size)

class TrashButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        color = QColor(148, 163, 184)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        rect = self.rect()
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        painter.drawRect(center_x - 6, center_y - 4, 12, 10)
        painter.drawLine(center_x - 8, center_y - 6, center_x + 8, center_y - 6)
        painter.drawLine(center_x - 3, center_y - 8, center_x + 3, center_y - 8)
        painter.drawLine(center_x - 2, center_y - 4, center_x - 2, center_y + 4)
        painter.drawLine(center_x, center_y - 4, center_x, center_y + 4)
        painter.drawLine(center_x + 2, center_y - 4, center_x + 2, center_y + 4)

class PlddtChartWidget(QWidget):
    def __init__(self, plddt_data, parent=None):
        super().__init__(parent)
        self.plddt_data = plddt_data or []
        self.setMinimumSize(220, 140)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        padding = 22
        if len(self.plddt_data) < 2:
            painter.setPen(QColor("#64748b"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No pLDDT data")
            return
        
        width = rect.width() - padding * 2
        height = rect.height() - padding * 2
        if width <= 0 or height <= 0:
            return
        
        title_font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#1e293b"))
        painter.drawText(padding, 14, "Per-residue pLDDT")
        
        axis_font = QFont("Arial", 8)
        painter.setFont(axis_font)
        painter.setPen(QColor("#64748b"))
        painter.drawText(padding, rect.height() - 4, "Residue Index")
        painter.save()
        painter.translate(10, padding + height / 2)
        painter.rotate(-90)
        painter.drawText(0, 0, "pLDDT")
        painter.restore()
        
        pen_axis = QPen(QColor("#e2e8f0"), 1)
        painter.setPen(pen_axis)
        painter.drawRect(padding, padding, width, height)
        
        tick_font = QFont("Arial", 8)
        painter.setFont(tick_font)
        painter.setPen(QColor("#94a3b8"))
        for val in [0, 50, 100]:
            y = padding + (height * (100 - val) / 100)
            painter.drawLine(padding - 4, int(y), padding, int(y))
            painter.drawText(2, int(y) + 3, str(val))
        
        x_labels = [1, max(1, len(self.plddt_data) // 2), len(self.plddt_data)]
        for val in x_labels:
            x = padding + (width * (val - 1) / max(1, len(self.plddt_data) - 1))
            painter.drawLine(int(x), padding + height, int(x), padding + height + 4)
            painter.drawText(int(x) - 6, padding + height + 14, str(val))
        
        pen_line = QPen(QColor("#3b82f6"), 1.5)
        painter.setPen(pen_line)
        n = len(self.plddt_data)
        for i in range(n - 1):
            v1 = max(0, min(100, self.plddt_data[i]))
            v2 = max(0, min(100, self.plddt_data[i + 1]))
            x1 = padding + (width * i / (n - 1))
            y1 = padding + (height * (100 - v1) / 100)
            x2 = padding + (width * (i + 1) / (n - 1))
            y2 = padding + (height * (100 - v2) / 100)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

class InfoButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self.info_text = ""
        self.clicked.connect(self.show_info)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def set_info_text(self, text):
        self.info_text = text
        self.setToolTip(text)
        
    def show_info(self):
        if self.info_text:
            QMessageBox.information(self.window(), "Parameters", self.info_text)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # 绘制圆圈
        circle_color = QColor(100, 116, 139)  # #64748b
        painter.setPen(QPen(circle_color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect.center(), 8, 8)
        
        # 绘制"i"字符
        painter.setPen(circle_color)
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        text_rect = painter.fontMetrics().boundingRect("i")
        text_x = center_x - text_rect.width() / 2
        text_y = center_y + text_rect.height() / 4
        painter.drawText(int(text_x), int(text_y), "i")

class PasteableTableWidget(QTableWidget):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_from_clipboard()
        else:
            super().keyPressEvent(event)

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        rows = [row for row in text.split('\n') if row]
        if not rows:
            return

        current_row = self.currentRow()
        current_col = self.currentColumn()

        if current_row < 0:
            current_row = self.rowCount() - 1 if self.rowCount() > 0 else 0
        if current_col < 0:
            current_col = 0

        while current_row + len(rows) > self.rowCount():
            row_idx = self.rowCount()
            self.insertRow(row_idx)
            
            type_combo = QComboBox()
            type_combo.addItems(["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"])
            type_combo.setMinimumWidth(120)
            type_combo.setFixedHeight(28)
            self.setCellWidget(row_idx, 1, type_combo)
            
            count_spin = QLineEdit("1")
            count_spin.setMinimumWidth(80)
            count_spin.setFixedHeight(28)
            self.setCellWidget(row_idx, 3, count_spin)

        for i, row_data in enumerate(rows):
            cells = row_data.split('\t')
            for j, cell_data in enumerate(cells):
                target_row = current_row + i
                target_col = current_col + j
                
                if target_col >= self.columnCount():
                    continue

                widget = self.cellWidget(target_row, target_col)
                if isinstance(widget, QComboBox):
                    idx = widget.findText(cell_data.strip())
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QLineEdit):
                    widget.setText(cell_data.strip())
                else:
                    item = self.item(target_row, target_col)
                    if not item:
                        item = QTableWidgetItem()
                        self.setItem(target_row, target_col, item)
                    item.setText(cell_data.strip())

class BatchPredictionWidget(QWidget):
    def __init__(self, parent=None, show_close=False, close_callback=None):
        super().__init__(parent)
        self.show_close = show_close
        self.close_callback = close_callback
        self.auto_save_dir = os.getcwd()
        self.create_ui()
        
    def create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(12)
        
        # 说明
        desc = QLabel("Create multiple prediction jobs using a table format. You can:")
        desc.setStyleSheet("color: #64748b;")
        layout.addWidget(desc)
        
        # 功能按钮
        btn_layout = QHBoxLayout()
        
        download_btn = QPushButton("📥 Download CSV Template")
        download_btn.setObjectName("OutlineButton")
        download_btn.clicked.connect(self.download_template)
        btn_layout.addWidget(download_btn)
        
        upload_btn = QPushButton("📤 Upload CSV")
        upload_btn.setObjectName("OutlineButton")
        upload_btn.clicked.connect(self.upload_csv)
        btn_layout.addWidget(upload_btn)
        
        add_row_btn = QPushButton("+ Add Row")
        add_row_btn.setObjectName("OutlineButton")
        add_row_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_row_btn)
        
        delete_rows_btn = QPushButton("- Delete Rows")
        delete_rows_btn.setObjectName("OutlineButton")
        delete_rows_btn.clicked.connect(self.delete_selected_rows)
        btn_layout.addWidget(delete_rows_btn)
        
        clear_btn = QPushButton("Clear Table")
        clear_btn.setObjectName("OutlineButton")
        clear_btn.clicked.connect(self.clear_table)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 提示信息 (单独占一行)
        hint_layout = QHBoxLayout()
        hint_label = QLabel(
            "💡 <b>Tip:</b> Support pasting from Excel. "
            "<b>Type constraints:</b> proteinChain / dnaSequence / rnaSequence / ligand / ion.<br>"
            "<b>Sequence constraints:</b> Protein (20 standard + X), DNA (A, T, G, C, N), RNA (A, U, G, C, N). "
            "<a href='https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md'>Format Spec</a>"
        )
        hint_label.setStyleSheet("color: #64748b; font-size: 11px;")
        hint_label.setTextFormat(Qt.TextFormat.RichText)
        hint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        hint_label.setOpenExternalLinks(True)
        hint_layout.addWidget(hint_label)
        
        self.details_btn = QPushButton("Details")
        self.details_btn.setObjectName("OutlineButton")
        self.details_btn.setStyleSheet("font-size: 11px; padding: 4px 10px;")
        self.details_btn.clicked.connect(self.toggle_details)
        hint_layout.addWidget(self.details_btn)
        
        hint_layout.addStretch()
        layout.addLayout(hint_layout)
        
        self.details_label = QLabel(
            "<b>Modifications:</b> Use Mod Type + Mod Position (protein) or Base Position (DNA/RNA). "
            "Multiple modifications supported with ';' separated values.<br>"
            "<b>Constraint:</b> JSON object or list, stored under constraint for the sequence.<br>"
            "<b>Pocket Constraint:</b> JSON object or list, stored under pocket_constraint for the sequence.<br>"
            "<b>Covalent Bonds:</b> Fill Bond Entity/Atom/Position columns; rows with the same Name are merged. "
            "<a href='https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md#covalent_bonds'>Tips</a>"
        )
        self.details_label.setStyleSheet("color: #475569; font-size: 11px; padding: 6px 0;")
        self.details_label.setTextFormat(Qt.TextFormat.RichText)
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.details_label.setOpenExternalLinks(True)
        self.details_label.setVisible(False)
        layout.addWidget(self.details_label)
        
        # 表格
        self.table = PasteableTableWidget()
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget QComboBox, QTableWidget QLineEdit {
                margin: 0px;
            }
            QHeaderView::section {
                background-color: #f1f5f9;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                font-weight: bold;
            }
        """)
        self.setup_table_columns()
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        table_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table_scroll.setWidget(self.table)
        layout.addWidget(table_scroll)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        if self.show_close and self.close_callback:
            close_btn = QPushButton("Close")
            close_btn.setObjectName("OutlineButton")
            close_btn.setFixedSize(100, 36)
            close_btn.clicked.connect(self.close_callback)
            bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)

    def toggle_details(self):
        self.details_label.setVisible(not self.details_label.isVisible())
        
    def setup_table_columns(self):
        columns = [
            "Name",
            "Type",
            "Sequence",
            "Count",
            "Mod Type",
            "Mod Position",
            "Base Position",
            "Constraint",
            "Pocket Constraint",
            "MSA Path (paired)",
            "MSA Path (unpaired)",
            "Templates Path",
            "Bond Entity1",
            "Bond Atom1",
            "Bond Position1",
            "Bond Entity2",
            "Bond Atom2",
            "Bond Position2"
        ]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(80)
        header.setFixedHeight(44)
        
        v_header = self.table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        v_header.setDefaultSectionSize(36)
        
        self.add_row()
        
    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        type_combo = QComboBox()
        type_combo.addItems(["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"])
        type_combo.setMinimumWidth(120)
        type_combo.setFixedHeight(28)
        self.table.setCellWidget(row, 1, type_combo)
        
        count_spin = QLineEdit("1")
        count_spin.setMinimumWidth(80)
        count_spin.setFixedHeight(28)
        self.table.setCellWidget(row, 3, count_spin)

    def delete_selected_rows(self):
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            QMessageBox.information(self, "Info", "Please select at least one cell in the rows you want to delete.")
            return
            
        rows_to_delete = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows_to_delete.add(row)
                
        # Sort in reverse order to not mess up indices during deletion
        for row in sorted(list(rows_to_delete), reverse=True):
            self.table.removeRow(row)
            
        # Ensure at least one row remains
        if self.table.rowCount() == 0:
            self.add_row()
        
    def clear_table(self):
        reply = QMessageBox.question(
            self, 
            "Clear Table", 
            "Are you sure you want to clear all data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)
            self.add_row()
            
    def download_template(self):
        import csv
        default_dir = os.getcwd()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save CSV Template", 
            os.path.join(default_dir, "batch_template.csv"), 
            "CSV Files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        
        if not file_path:
            return
            
        try:
            headers = [
                "Name",
                "Type",
                "Sequence",
                "Count",
                "Mod Type",
                "Mod Position",
                "Base Position",
                "Constraint",
                "Pocket Constraint",
                "MSA Path (paired)",
                "MSA Path (unpaired)",
                "Templates Path",
                "Bond Entity1",
                "Bond Atom1",
                "Bond Position1",
                "Bond Entity2",
                "Bond Atom2",
                "Bond Position2"
            ]
            
            example_rows = [
                ["job_1", "proteinChain", "PREACHINGS", "1", "CCD_HY3", "1", "", "", "", "", "", "", "A", "C1", "1", "B", "C2", "2"],
                ["job_1", "ligand", "C1CCO1", "1", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["job_2", "dnaSequence", "GATTACA", "1", "CCD_6OG", "", "1", "", "", "", "", "", "", "", "", "", "", ""]
            ]
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(example_rows)
                
            QMessageBox.information(self, "Success", f"Template saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save template:\n{str(e)}\n\nPlease try saving to a different location like your Desktop.")
            
    def upload_csv(self):
        import csv
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Upload CSV", 
            "", 
            "CSV Files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            if len(rows) < 2:
                QMessageBox.warning(self, "Error", "CSV file must have at least a header and one data row.")
                return
                
            self.table.setRowCount(0)
            
            for row_data in rows[1:]:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                for col in range(min(18, len(row_data))):
                    if col == 1:
                        type_combo = QComboBox()
                        type_combo.addItems(["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"])
                        type_combo.setMinimumWidth(180)
                        type_combo.setMinimumHeight(28)
                        idx = type_combo.findText(row_data[col])
                        if idx >= 0:
                            type_combo.setCurrentIndex(idx)
                        self.table.setCellWidget(row, col, type_combo)
                    elif col == 3:
                        count_spin = QLineEdit(row_data[col] or "1")
                        count_spin.setMinimumWidth(100)
                        count_spin.setMinimumHeight(28)
                        self.table.setCellWidget(row, col, count_spin)
                    else:
                        item = QTableWidgetItem(row_data[col])
                        self.table.setItem(row, col, item)
                        
            QMessageBox.information(self, "Success", f"Loaded {len(rows)-1} rows from CSV.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load CSV:\n{str(e)}")
            
    def get_batch_jobs(self):
        jobs = []
        jobs_by_name = {}
        errors = []
        
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            type_widget = self.table.cellWidget(row, 1)
            seq_item = self.table.item(row, 2)
            count_widget = self.table.cellWidget(row, 3)
            mod_type_item = self.table.item(row, 4)
            mod_pos_item = self.table.item(row, 5)
            base_pos_item = self.table.item(row, 6)
            constraint_item = self.table.item(row, 7)
            pocket_constraint_item = self.table.item(row, 8)
            msa_paired_item = self.table.item(row, 9)
            msa_unpaired_item = self.table.item(row, 10)
            templates_item = self.table.item(row, 11)
            bond_entity1_item = self.table.item(row, 12)
            bond_atom1_item = self.table.item(row, 13)
            bond_pos1_item = self.table.item(row, 14)
            bond_entity2_item = self.table.item(row, 15)
            bond_atom2_item = self.table.item(row, 16)
            bond_pos2_item = self.table.item(row, 17)
            
            if not name_item or not name_item.text().strip():
                continue
                
            if not type_widget:
                continue
                
            seq_type = type_widget.currentText()
            raw_sequence = seq_item.text() if seq_item else ""
            
            is_ligand_or_ion = seq_type in ["ligand", "ion"]
            cleaned_sequence = normalize_sequence_text(raw_sequence, is_ligand_or_ion)
            if seq_item and cleaned_sequence != raw_sequence.strip() and not is_ligand_or_ion:
                seq_item.setText(cleaned_sequence)
            sequence = cleaned_sequence.strip()
            if not sequence:
                errors.append(f"Row {row + 1}: sequence is required")
                continue
            
            seq_error = validate_sequence_by_type(sequence, seq_type, f"Row {row + 1}")
            if seq_error:
                errors.append(seq_error)
                continue
            count = int(count_widget.text()) if count_widget else 1
            
            sequence_data = {}
            
            if seq_type == "proteinChain":
                sequence_data = {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": count
                    }
                }
            elif seq_type == "dnaSequence":
                sequence_data = {
                    "dnaSequence": {
                        "sequence": sequence,
                        "count": count
                    }
                }
            elif seq_type == "rnaSequence":
                sequence_data = {
                    "rnaSequence": {
                        "sequence": sequence,
                        "count": count
                    }
                }
            elif seq_type == "ligand":
                sequence_data = {
                    "ligand": {
                        "ligand": sequence,
                        "count": count
                    }
                }
            elif seq_type == "ion":
                sequence_data = {
                    "ion": {
                        "ion": sequence,
                        "count": count
                    }
                }
                
            mod_type_text = mod_type_item.text().strip() if mod_type_item else ""
            mod_pos_text = mod_pos_item.text().strip() if mod_pos_item else ""
            base_pos_text = base_pos_item.text().strip() if base_pos_item else ""
            
            if mod_type_text or mod_pos_text or base_pos_text:
                mod_types = [t.strip() for t in mod_type_text.split(";") if t.strip()] if mod_type_text else []
                mod_positions = [p.strip() for p in mod_pos_text.split(";") if p.strip()] if mod_pos_text else []
                base_positions = [p.strip() for p in base_pos_text.split(";") if p.strip()] if base_pos_text else []
                
                if not mod_types:
                    errors.append(f"Row {row + 1}: Mod Type is required when modification fields are provided")
                else:
                    if seq_type == "proteinChain":
                        if not mod_positions or len(mod_positions) != len(mod_types):
                            errors.append(f"Row {row + 1}: Mod Position count mismatch")
                        else:
                            mods = []
                            for i in range(len(mod_types)):
                                mods.append({"ptmType": mod_types[i], "ptmPosition": int(mod_positions[i])})
                            sequence_data["proteinChain"]["modifications"] = mods
                    elif seq_type in ["dnaSequence", "rnaSequence"]:
                        if not base_positions or len(base_positions) != len(mod_types):
                            errors.append(f"Row {row + 1}: Base Position count mismatch")
                        else:
                            mods = []
                            for i in range(len(mod_types)):
                                mods.append({"modificationType": mod_types[i], "basePosition": int(base_positions[i])})
                            sequence_data[seq_type]["modifications"] = mods
                    else:
                        errors.append(f"Row {row + 1}: Modifications only supported for protein/dna/rna")
            
            if constraint_item and constraint_item.text().strip():
                try:
                    constraint_val = json.loads(constraint_item.text())
                    sequence_data[seq_type]["constraint"] = constraint_val
                except Exception:
                    errors.append(f"Row {row + 1}: invalid Constraint JSON")
            
            if pocket_constraint_item and pocket_constraint_item.text().strip():
                try:
                    pocket_val = json.loads(pocket_constraint_item.text())
                    sequence_data[seq_type]["pocket_constraint"] = pocket_val
                except Exception:
                    errors.append(f"Row {row + 1}: invalid Pocket Constraint JSON")
                    
            if msa_paired_item and msa_paired_item.text().strip():
                if seq_type == "proteinChain":
                    sequence_data["proteinChain"]["pairedMsaPath"] = msa_paired_item.text().strip()
                    
            if msa_unpaired_item and msa_unpaired_item.text().strip():
                if seq_type == "proteinChain":
                    sequence_data["proteinChain"]["unpairedMsaPath"] = msa_unpaired_item.text().strip()
                    
            if templates_item and templates_item.text().strip():
                if seq_type == "proteinChain":
                    sequence_data["proteinChain"]["templatesPath"] = templates_item.text().strip()
            
            job_name = name_item.text().strip()
            if job_name not in jobs_by_name:
                jobs_by_name[job_name] = {
                    "name": job_name,
                    "sequences": []
                }
            jobs_by_name[job_name]["sequences"].append(sequence_data)
            
            bond_entity1 = bond_entity1_item.text().strip() if bond_entity1_item else ""
            bond_atom1 = bond_atom1_item.text().strip() if bond_atom1_item else ""
            bond_pos1 = bond_pos1_item.text().strip() if bond_pos1_item else ""
            bond_entity2 = bond_entity2_item.text().strip() if bond_entity2_item else ""
            bond_atom2 = bond_atom2_item.text().strip() if bond_atom2_item else ""
            bond_pos2 = bond_pos2_item.text().strip() if bond_pos2_item else ""
            
            bond_fields = [bond_entity1, bond_atom1, bond_pos1, bond_entity2, bond_atom2, bond_pos2]
            if any(bond_fields):
                if not all(bond_fields):
                    errors.append(f"Row {row + 1}: incomplete Covalent Bond fields")
                else:
                    try:
                        bond = {
                            "entity1": bond_entity1,
                            "atom1": bond_atom1,
                            "position1": int(bond_pos1),
                            "entity2": bond_entity2,
                            "atom2": bond_atom2,
                            "position2": int(bond_pos2)
                        }
                        if "covalent_bonds" not in jobs_by_name[job_name]:
                            jobs_by_name[job_name]["covalent_bonds"] = []
                        jobs_by_name[job_name]["covalent_bonds"].append(bond)
                    except Exception:
                        errors.append(f"Row {row + 1}: invalid Covalent Bond positions")
        
        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(errors))
            return None
        
        jobs = list(jobs_by_name.values())
            
        if not jobs:
            QMessageBox.warning(self, "Warning", "No valid job data to generate.")
            return None
        
        return jobs
            
    def generate_batch_json(self):
        jobs = self.get_batch_jobs()
        if not jobs:
            return
            
        try:
            default_name = "batch_predictions.json"
            try:
                parent = self.parent()
                while parent and not hasattr(parent, "inp_name"):
                    parent = parent.parent()
                if parent and hasattr(parent, "inp_name"):
                    task_name = parent.inp_name.text().strip()
                    if task_name:
                        out_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs", task_name))
                        os.makedirs(out_dir, exist_ok=True)
                        file_path = os.path.join(out_dir, default_name)
                    else:
                        file_path = os.path.join(self.auto_save_dir, default_name)
                else:
                    file_path = os.path.join(self.auto_save_dir, default_name)
            except Exception:
                file_path = os.path.join(self.auto_save_dir, default_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2)
            
            # Open the file with the default system application
                try:
                    if platform.system() == 'Windows':
                        os.startfile(file_path)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.Popen(['open', file_path])
                    else:  # Linux
                        subprocess.Popen(['xdg-open', file_path])
                except Exception as open_e:
                    print(f"Failed to open file automatically: {open_e}")
                    
                QMessageBox.information(self, "Success", f"Batch JSON generated and opened:\n{file_path}\n\nGenerated {len(jobs)} jobs.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save JSON:\n{str(e)}")
        
class ExpandableTaskWidget(QFrame):
    def __init__(self, task_data, parent=None):
        super().__init__(parent)
        self.task_data = task_data
        self.is_expanded = False
        self.setStyleSheet("""
            ExpandableTaskWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #ffffff;
                margin: 2px 0;
            }
        """)
        self.create_ui()
        
    def create_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(6)
        
        # 任务标题行
        header_layout = QHBoxLayout()
        
        # 复选框：批量选择
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        header_layout.addWidget(self.checkbox)
        
        # 展开/折叠按钮
        self.toggle_btn = ArrowButton()
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_expand)
        header_layout.addWidget(self.toggle_btn)
        
        # 任务名称
        task_name_label = QLabel(self.task_data['name'])
        task_name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        task_name_label.setStyleSheet("color: #1e293b;")
        header_layout.addWidget(task_name_label)
        
        # 状态
        status_label = QLabel(self.task_data['status'])
        if self.task_data['status'] == 'Failed':
            status_label.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 11px;")
        else:
            status_label.setStyleSheet("color: #16a34a; font-weight: 600; font-size: 11px;")
        header_layout.addWidget(status_label)
        
        header_layout.addStretch()
        
        # Input按钮
        input_btn = QPushButton("Input file")
        input_btn.setObjectName("PrimaryButton")
        input_btn.setFixedSize(75, 28)
        input_btn.setStyleSheet("font-size: 11px; padding: 2px 10px; font-weight: 600;")
        input_btn.clicked.connect(lambda: self.view_input())
        header_layout.addWidget(input_btn)
        
        # Output按钮
        output_btn = QPushButton("Output files")
        output_btn.setObjectName("PrimaryButton")
        output_btn.setFixedSize(90, 28)
        output_btn.setStyleSheet("font-size: 11px; padding: 2px 10px; font-weight: 600;")
        output_btn.clicked.connect(lambda: self.open_output())
        header_layout.addWidget(output_btn)
        
        # 如果有ERR，添加Error log按钮
        if self.task_data.get('has_err'):
            error_log_btn = QPushButton("Error log")
            error_log_btn.setStyleSheet("""
                QPushButton {
                    background-color: #fef2f2;
                    color: #dc2626;
                    border: 1px solid #fecaca;
                    border-radius: 14px;
                    font-size: 11px;
                    padding: 2px 10px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #fee2e2;
                }
            """)
            error_log_btn.setFixedSize(75, 28)
            error_log_btn.clicked.connect(lambda: self.view_error_log())
            header_layout.addWidget(error_log_btn)
        
        # View all按钮（只有有结果时显示）
        if self.task_data['samples']:
            view_all_btn = QPushButton("View all in PyMOL")
            view_all_btn.setObjectName("OutlineButton")
            view_all_btn.setFixedSize(120, 28)
            view_all_btn.setStyleSheet("font-size: 11px; padding: 2px 10px; font-weight: 600;")
            view_all_btn.clicked.connect(lambda: self.view_all_structures())
            header_layout.addWidget(view_all_btn)
            
        # Delete 按钮
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #fef2f2;
                color: #ef4444;
                border: 1px solid #fecaca;
                border-radius: 4px;
                font-size: 11px;
                padding: 2px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #fee2e2;
            }
        """)
        delete_btn.setFixedSize(65, 28)
        delete_btn.clicked.connect(self.delete_task)
        header_layout.addWidget(delete_btn)
        
        self.main_layout.addLayout(header_layout)
        
        # 展开内容区域
        self.expanded_content = QWidget()
        self.expanded_layout = QVBoxLayout(self.expanded_content)
        self.expanded_layout.setContentsMargins(28, 4, 0, 4)
        self.expanded_layout.setSpacing(2)
        self.expanded_content.setVisible(False)
        
        # 加载样本列表
        self.load_samples()
        
        self.main_layout.addWidget(self.expanded_content)
        
    def load_samples(self):
        # 清除现有内容
        for i in reversed(range(self.expanded_layout.count())):
            self.expanded_layout.itemAt(i).widget().setParent(None)
        
        # 添加样本
        for sample in self.task_data['samples']:
            sample_widget = self.create_sample_widget(sample)
            self.expanded_layout.addWidget(sample_widget)
            
    def create_sample_widget(self, sample):
        frame = QFrame()
        frame.sample_data = sample
        frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 4px;
                padding: 4px;
                margin: 1px 0;
            }
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        
        # 样本选择复选框
        checkbox = QCheckBox()
        checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """)
        frame.checkbox = checkbox
        layout.addWidget(checkbox)
        
        # 样本名称
        sample_label = QLabel(sample['name'])
        sample_label.setFont(QFont("Arial", 10))
        sample_label.setStyleSheet("color: #475569;")
        layout.addWidget(sample_label, stretch=1)
        
        if sample['confidence']:
            # 将pTM和ipTM显示在一行
            conf_text = sample['confidence'].replace('\n', ', ')
            conf_label = QLabel(conf_text)
            conf_label.setStyleSheet("color: #64748b; font-size: 10px;")
            conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(conf_label)
        
        # 显示pLDDT和GPDE
        metrics_text = ""
        if sample.get('plddt') is not None:
            if isinstance(sample['plddt'], list):
                avg_plddt = sum(sample['plddt']) / len(sample['plddt']) if sample['plddt'] else 0
                metrics_text += f"pLDDT = {avg_plddt:.1f}  "
            else:
                metrics_text += f"pLDDT = {sample['plddt']:.1f}  "
        if sample.get('gpde') is not None:
            metrics_text += f"GPDE = {sample['gpde']:.2f}"
        if metrics_text:
            metrics_label = QLabel(metrics_text)
            metrics_label.setStyleSheet("color: #64748b; font-size: 10px;")
            metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(metrics_label)
        
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f9ff;
                color: #0369a1;
                border: 1px solid #bae6fd;
                border-radius: 11px;
                font-size: 10px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #e0f2fe;
            }
        """)
        preview_btn.setFixedSize(60, 22)
        preview_btn.clicked.connect(lambda: self.preview_sample(sample))
        layout.addWidget(preview_btn)
        
        view_btn = QPushButton("View in PyMOL")
        view_btn.setObjectName("SecondaryButton")
        view_btn.setFixedSize(110, 22)
        view_btn.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        view_btn.clicked.connect(lambda: self.view_structure(sample['cif_file']))
        layout.addWidget(view_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("""
            QPushButton {
                color: #ef4444;
                border: 1px solid #fecaca;
                border-radius: 4px;
                font-size: 10px;
                padding: 2px 6px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #fee2e2;
            }
        """)
        delete_btn.setFixedSize(50, 22)
        delete_btn.clicked.connect(lambda: self.delete_sample(sample, frame))
        layout.addWidget(delete_btn)
        
        return frame
        
    def toggle_expand(self):
        self.is_expanded = not self.is_expanded
        self.toggle_btn.set_expanded(self.is_expanded)
        self.expanded_content.setVisible(self.is_expanded)
        
    def view_input(self):
        input_path = os.path.join(self.task_data['dir'], "inputs.json")
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Not Found", f"inputs.json not found in:\n{self.task_data['dir']}")
            return
            
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Input JSON - {self.task_data['name']}")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        text_edit.setText(content)
        layout.addWidget(text_edit)
        dialog.exec()
        
    def open_output(self):
        if sys.platform == 'win32':
            os.startfile(self.task_data['dir'])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', self.task_data['dir']])
        else:
            subprocess.Popen(['xdg-open', self.task_data['dir']])
            
    def view_structure(self, cif_file):
        self.open_in_pymol(cif_file)

    def delete_task(self):
        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Deletion")
        dialog.resize(350, 150)
        
        layout = QVBoxLayout(dialog)
        
        msg_label = QLabel(f"Are you sure you want to delete the task '{self.task_data['name']}'?")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        delete_files_cb = QCheckBox("Move files to Trash")
        delete_files_cb.setStyleSheet("margin-top: 10px; margin-bottom: 10px; color: #ef4444;")
        layout.addWidget(delete_files_cb)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("Delete")
        confirm_btn.setStyleSheet("background-color: #ef4444; color: white; border: none; padding: 4px 12px; border-radius: 4px;")
        confirm_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(confirm_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                if delete_files_cb.isChecked() and os.path.exists(self.task_data['dir']):
                    safe_trash_delete(self.task_data['dir'])
                
                # 通知父组件刷新列表
                parent = self.parent()
                while parent and not hasattr(parent, 'refresh_expandable_history'):
                    parent = parent.parent()
                
                if parent and hasattr(parent, 'refresh_expandable_history'):
                    # 从全局列表中移除并刷新
                    if self.task_data in parent.prediction_history_tasks:
                        parent.prediction_history_tasks.remove(self.task_data)
                    parent.refresh_expandable_history()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete task:\n{str(e)}")

    def delete_sample(self, sample, frame_widget):
        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Deletion")
        dialog.resize(350, 150)
        
        layout = QVBoxLayout(dialog)
        
        msg_label = QLabel(f"Are you sure you want to delete the sample '{sample['name']}'?")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        delete_files_cb = QCheckBox("Move files to Trash")
        delete_files_cb.setStyleSheet("margin-top: 10px; margin-bottom: 10px; color: #ef4444;")
        layout.addWidget(delete_files_cb)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("Delete")
        confirm_btn.setStyleSheet("background-color: #ef4444; color: white; border: none; padding: 4px 12px; border-radius: 4px;")
        confirm_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(confirm_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                # 只有勾选了才物理删除文件
                if delete_files_cb.isChecked():
                    if sample['cif_file'] and os.path.exists(sample['cif_file']):
                        safe_trash_delete(sample['cif_file'])
                    if sample['json_file'] and os.path.exists(sample['json_file']):
                        safe_trash_delete(sample['json_file'])
                
                # 从数据中移除并销毁UI
                if sample in self.task_data['samples']:
                    self.task_data['samples'].remove(sample)
                frame_widget.setParent(None)
                frame_widget.deleteLater()
                
                # 如果没有样本了，考虑自动删除或更新状态
                if not self.task_data['samples']:
                    self.task_data['status'] = 'Empty'
                    # 更新状态标签
                    header_layout = self.main_layout.itemAt(0).layout()
                    # 状态标签现在是第3个（复选框、展开按钮、标题之后）
                    status_label = header_layout.itemAt(3).widget()
                    if isinstance(status_label, QLabel):
                        status_label.setText('Empty')
                        status_label.setStyleSheet("color: #64748b; font-weight: 600; font-size: 11px;")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete sample:\n{str(e)}")

    def open_structure_viewer(self, cif_file, title_text, plddt_data=None):
        if not WEB_ENGINE_AVAILABLE:
            QMessageBox.warning(
                self,
                "Viewer Unavailable",
                "3D viewer requires PyQt6-WebEngine.\nInstall: pip install PyQt6-WebEngine"
            )
            return
        
        try:
            with open(cif_file, 'r', encoding='utf-8') as f:
                cif_content = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read structure file:\n{str(e)}")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Structure Viewer - {title_text}")
        dialog.resize(900, 700)
        layout = QHBoxLayout(dialog)
        
        viewer = QWebEngineView()
        cif_js = json.dumps(cif_content)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <style>
                html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
                #viewer {{ width: 100%; height: 100%; }}
            </style>
        </head>
        <body>
            <div id="viewer"></div>
            <script>
                const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
                const cif = {cif_js};
                viewer.addModel(cif, "cif");
                
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorfunc: function(atom) {{
                            if (atom.b > 90) return '#0053d6';
                            if (atom.b > 70) return '#65cbf3';
                            if (atom.b > 50) return '#ffdb13';
                            return '#ff7d45';
                        }}
                    }}
                }});
                
                viewer.zoomTo();
                viewer.render();
            </script>
        </body>
        </html>
        """
        viewer.setHtml(html)
        side_panel = QWidget()
        side_panel.setFixedWidth(240)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(10)
        
        legend_title = QLabel("pLDDT (Confidence)")
        legend_title.setStyleSheet("font-weight: 600; color: #1e293b; font-size: 12px;")
        side_layout.addWidget(legend_title)
        
        def legend_row(color, text):
            row = QHBoxLayout()
            box = QFrame()
            box.setFixedSize(12, 12)
            box.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            label = QLabel(text)
            label.setStyleSheet("color: #475569; font-size: 11px;")
            row.addWidget(box)
            row.addSpacing(6)
            row.addWidget(label)
            row.addStretch()
            side_layout.addLayout(row)
        
        legend_row("#0053d6", "Very High (>90)")
        legend_row("#65cbf3", "High (70-90)")
        legend_row("#ffdb13", "Low (50-70)")
        legend_row("#ff7d45", "Very Low (<50)")
        
        plddt_values = None
        if plddt_data is not None:
            if not isinstance(plddt_data, list):
                plddt_data = [plddt_data]
            plddt_values = plddt_data
        if not plddt_values or len(plddt_values) < 2:
            plddt_values = extract_plddt_from_cif(cif_content)
        
        if plddt_values and len(plddt_values) > 1:
            chart = PlddtChartWidget(plddt_values)
            side_layout.addWidget(chart)
        else:
            empty_label = QLabel("No pLDDT data")
            empty_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
            side_layout.addWidget(empty_label)
        
        side_layout.addStretch()
        
        layout.addWidget(viewer, stretch=1)
        layout.addWidget(side_panel)
        dialog.exec()
        
    def view_all_structures(self):
        # 收集所有cif文件
        cif_files = []
        for sample in self.task_data['samples']:
            if sample['cif_file'] and os.path.exists(sample['cif_file']):
                cif_files.append(sample['cif_file'])
        
        if not cif_files:
            QMessageBox.warning(self, "Not Found", "No CIF files found for this task.")
            return
        
        # 用PyMOL打开所有文件
        self.open_all_in_pymol(cif_files)
        
    def open_in_pymol(self, cif_file):
        # 尝试不同的PyMOL路径
        pymol_paths = ['pymol', '/opt/homebrew/bin/pymol']
        opened = False
        
        for pymol_exec in pymol_paths:
            try:
                subprocess.Popen([pymol_exec, cif_file])
                opened = True
                break
            except Exception as e:
                print(f"Failed to open with {pymol_exec}: {e}")
        
        if not opened:
            QMessageBox.warning(self, "Error", "Could not open PyMOL. Please make sure PyMOL is installed and in your PATH.")
            
    def open_all_in_pymol(self, cif_files):
        # 尝试不同的PyMOL路径
        pymol_paths = ['pymol', '/opt/homebrew/bin/pymol']
        opened = False
        
        for pymol_exec in pymol_paths:
            try:
                subprocess.Popen([pymol_exec] + cif_files)
                opened = True
                break
            except Exception as e:
                print(f"Failed to open with {pymol_exec}: {e}")
        
        if not opened:
            QMessageBox.warning(self, "Error", "Could not open PyMOL. Please make sure PyMOL is installed and in your PATH.")
            
    def view_error_log(self):
        err_dir = self.task_data.get('err_dir')
        if not err_dir or not os.path.exists(err_dir):
            QMessageBox.warning(self, "Not Found", "ERR directory not found.")
            return
            
        # 查找所有txt文件
        txt_files = glob.glob(os.path.join(err_dir, "*.txt"))
        
        if not txt_files:
            QMessageBox.information(self, "Info", "No log files found in ERR directory.")
            return
            
        # 读取所有文件内容
        log_content = ""
        for txt_file in sorted(txt_files):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    log_content += f"=== {os.path.basename(txt_file)} ===\n\n"
                    log_content += content
                    log_content += "\n\n"
            except Exception as e:
                log_content += f"Failed to read {os.path.basename(txt_file)}: {e}\n\n"
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Error Log - {self.task_data['name']}")
        dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        text_edit.setText(log_content)
        layout.addWidget(text_edit)
        
        dialog.exec()
            
    def preview_sample(self, sample):
        if sample['cif_file'] and os.path.exists(sample['cif_file']):
            self.open_structure_viewer(sample['cif_file'], sample['name'], sample.get('plddt'))
        else:
            QMessageBox.warning(self, "Not Found", "Structure file not found.")

class ToggleGroup(QWidget):
    def __init__(self, default_val=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.btn_true = QPushButton("true")
        self.btn_true.setObjectName("ToggleLeft")
        self.btn_true.setCheckable(True)
        self.btn_true.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        self.btn_false = QPushButton("false")
        self.btn_false.setObjectName("ToggleRight")
        self.btn_false.setCheckable(True)
        self.btn_false.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        self.btn_true.setChecked(default_val)
        self.btn_false.setChecked(not default_val)
        
        self.btn_true.clicked.connect(lambda: self.btn_false.setChecked(False) if self.btn_true.isChecked() else self.btn_true.setChecked(True))
        self.btn_false.clicked.connect(lambda: self.btn_true.setChecked(False) if self.btn_false.isChecked() else self.btn_false.setChecked(True))
        
        layout.addWidget(self.btn_true)
        layout.addWidget(self.btn_false)
        layout.addStretch()

    def get_value(self):
        return self.btn_true.isChecked()

# --- 主窗口视图 ---
class ProtenixServerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protenix GUI")
        self.resize(1200, 850)
        self.setStyleSheet(STYLE_SHEET)
        
        # 动态历史数据
        self.prediction_history_tasks = []
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.create_sidebar(main_layout)
        
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget, stretch=1)
        
        self.page_add_prediction = self.create_add_prediction_page()
        self.page_prediction_history = self.create_expandable_history_page("Predictions")
        
        self.stacked_widget.addWidget(self.page_add_prediction)
        self.stacked_widget.addWidget(self.page_prediction_history)
        
        # 启动时默认加载当前目录下的 outputs 文件夹
        default_outputs_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
        if os.path.exists(default_outputs_dir) and os.path.isdir(default_outputs_dir):
            self.load_history_from_dir(default_outputs_dir, show_msg=False)
        
    def create_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(210)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title_label = QLabel("📊 Protenix GUI")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #0f172a; margin-bottom: 20px;")
        sidebar_layout.addWidget(title_label)
        
        self.nav_buttons = []
        
        def add_nav_section(title, items):
            section_label = QLabel(title)
            section_label.setStyleSheet("color: #64748b; font-weight: bold; font-size: 12px; margin-top: 15px; margin-bottom: 5px;")
            sidebar_layout.addWidget(section_label)
            
            for icon, text, index in items:
                btn = QPushButton(f"{icon}  {text}")
                btn.setObjectName("SidebarButton")
                btn.setCheckable(True)
                btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                btn.clicked.connect(lambda checked, idx=index, b=btn: self.switch_page(idx, b))
                sidebar_layout.addWidget(btn)
                self.nav_buttons.append(btn)
        
        add_nav_section("🧬 Structure Prediction", [
            ("➕", "Add Prediction", 0),
            ("🕒", "Prediction History", 1)
        ])
        
        sidebar_layout.addStretch()
        
        parent_layout.addWidget(sidebar)
        
        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

    def switch_page(self, index, clicked_button):
        self.stacked_widget.setCurrentIndex(index)
        for btn in self.nav_buttons:
            if btn != clicked_button:
                btn.setChecked(False)
        if index == 1:
            self.refresh_expandable_history()

    def create_form_row(self, label_text, widget, is_required=False):
        layout = QHBoxLayout()
        lbl = QLabel(f"<span style='color:red;'>*</span> {label_text}" if is_required else label_text)
        lbl.setFixedWidth(130) # 增加宽度，避免被遮挡
        layout.addWidget(lbl)
        layout.addWidget(widget)
        layout.addStretch()
        return layout

    def browse_script(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Protenix Entry Script", 
            "", 
            "Python Files (*.py);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if fname:
            self.inp_script.setText(fname)

    def browse_cuda(self):
        dname = QFileDialog.getExistingDirectory(
            self, 
            "Select CUDA Root Directory",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if dname:
            self.inp_cuda_home.setText(dname)
            
    def browse_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            "",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if dir_path:
            line_edit.setText(dir_path)

    # --- 页面 1: Add Prediction ---
    def create_add_prediction_page(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        container = QWidget()
        container.setMinimumWidth(600)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # --- Global Card ---
        card_global = Card()
        global_header = QHBoxLayout()
        global_title = QLabel("Global")
        global_title.setObjectName("CardTitle")
        global_header.addWidget(global_title)
        global_header.addStretch()
        
        load_json_btn = QPushButton("Load JSON")
        load_json_btn.setObjectName("OutlineButton")
        load_json_btn.setStyleSheet("font-size: 11px; padding: 4px 12px; margin-bottom: 8px; margin-right: 8px;")
        load_json_btn.clicked.connect(self.load_json_to_ui)
        global_header.addWidget(load_json_btn)
        
        new_pred_btn = QPushButton("New Prediction")
        new_pred_btn.setObjectName("OutlineButton")
        new_pred_btn.setStyleSheet("font-size: 11px; padding: 4px 12px; margin-bottom: 8px;")
        new_pred_btn.clicked.connect(self.reset_all_inputs)
        global_header.addWidget(new_pred_btn)
        
        card_global.add_layout(global_header)
        
        self.inp_name = QLineEdit("protenix_prediction_job_1")
        self.inp_name.setMinimumWidth(120)
        self.inp_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card_global.add_layout(self.create_form_row("Task name", self.inp_name, True))
        
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "protenix-v2",
            "protenix_base_default_v1.0.0",
            "protenix_base_20250630_v1.0.0",
            "protenix_base_default_v0.5.0",
            "protenix_base_constraint_v0.5.0",
            "protenix_mini_esm_v0.5.0",
            "protenix_mini_ism_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_tiny_default_v0.5.0"
        ])
        self.combo_model.setMinimumWidth(140)
        self.combo_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.combo_model)
        model_link = QLabel('<a href="https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md">Supported models</a>')
        model_link.setOpenExternalLinks(True)
        model_link.setStyleSheet("color: #2563eb; font-size: 11px;")
        model_layout.addSpacing(8)
        model_layout.addWidget(model_link)
        model_layout.addStretch()
        card_global.add_layout(self.create_form_row("Model", model_container))

        self.combo_device = QComboBox()
        self.combo_device.addItems(["GPU (CUDA)", "CPU"])
        self.combo_device.setMinimumWidth(120)
        self.combo_device.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card_global.add_layout(self.create_form_row("Device", self.combo_device, True))

        self.inp_script = "protenix"

        cuda_container = QWidget()
        cuda_layout = QHBoxLayout(cuda_container)
        cuda_layout.setContentsMargins(0, 0, 0, 0)
        
        default_cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        self.inp_cuda_home = QLineEdit(default_cuda)
        self.inp_cuda_home.setMinimumWidth(120)
        self.inp_cuda_home.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inp_cuda_home.setPlaceholderText("e.g. /usr/local/cuda-11.8")
        cuda_layout.addWidget(self.inp_cuda_home)
        
        browse_cuda_btn = QPushButton("Browse...")
        browse_cuda_btn.clicked.connect(self.browse_cuda)
        cuda_layout.addWidget(browse_cuda_btn)
        cuda_layout.addStretch()
        card_global.add_layout(self.create_form_row("CUDA_HOME", cuda_container, False))

        # 将三个toggle并排放在一行
        toggles_layout = QHBoxLayout()
        
        toggles_layout.addWidget(QLabel("Use MSA"))
        self.toggle_msa = ToggleGroup(default_val=True)
        toggles_layout.addWidget(self.toggle_msa)
        
        toggles_layout.addSpacing(20)
        
        toggles_layout.addWidget(QLabel("Use template"))
        self.toggle_template = ToggleGroup(default_val=False)
        toggles_layout.addWidget(self.toggle_template)
        
        toggles_layout.addSpacing(20)
        
        toggles_layout.addWidget(QLabel("Use RNA MSA"))
        self.toggle_rna_msa = ToggleGroup(default_val=False)
        toggles_layout.addWidget(self.toggle_rna_msa)
        
        toggles_layout.addStretch()
        card_global.add_layout(toggles_layout)
        
        msa_dir_container = QWidget()
        msa_dir_layout = QHBoxLayout(msa_dir_container)
        msa_dir_layout.setContentsMargins(0, 0, 0, 0)
        
        self.inp_msa_dir = QLineEdit()
        self.inp_msa_dir.setPlaceholderText("Directory containing pre-computed MSA and Template features")
        msa_dir_layout.addWidget(self.inp_msa_dir)
        browse_msa_btn = QPushButton("Browse...")
        browse_msa_btn.clicked.connect(lambda: self.browse_directory(self.inp_msa_dir))
        msa_dir_layout.addWidget(browse_msa_btn)
        
        card_global.add_layout(self.create_form_row("MSA/Template search dir", msa_dir_container))
        
        # 将提示说明单独放在 Global Card 的一行，使其可以延伸至边缘
        msa_hint = QLabel("<i>For protein, 'pairing' or 'non_pairing' a3m/msa files will be used as MSA, and 'concat.hhr' or 'hmmsearch.a3m' will be used as Template. For RNA, any single a3m/msa file found will be used. Otherwise, please select manually.</i>")
        msa_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        msa_hint.setWordWrap(True)
        card_global.add_widget(msa_hint)
        
        card_global.layout.setContentsMargins(12, 8, 12, 8)
        card_global.layout.setSpacing(8)
        layout.addWidget(card_global)
        
        single_scroll = QScrollArea()
        single_scroll.setWidgetResizable(True)
        single_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        single_container = QWidget()
        single_container.setMinimumWidth(560)
        single_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        single_layout = QVBoxLayout(single_container)
        single_layout.setContentsMargins(8, 6, 8, 8)
        single_layout.setSpacing(14)
        
        seq_section = QVBoxLayout()
        seq_header = QHBoxLayout()
        seq_title = QLabel("Sequences")
        seq_title.setStyleSheet("font-size: 14px; font-weight: 400; color: #1e293b;")
        seq_header.addWidget(seq_title)
        add_seq_btn = QPushButton("+ Add sequence")
        add_seq_btn.setObjectName("OutlineButton")
        add_seq_btn.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        add_seq_btn.clicked.connect(self.add_sequence)
        seq_header.addWidget(add_seq_btn)
        seq_header.addStretch()
        seq_section.addLayout(seq_header)
        
        self.seqs_container = QWidget()
        self.seqs_layout = QVBoxLayout(self.seqs_container)
        self.seqs_layout.setContentsMargins(0, 0, 0, 0)
        self.seqs_layout.setSpacing(10)
        self.sequences = []
        self.seqs_container.setMinimumWidth(520)
        self.seqs_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        self.add_sequence()
        seq_section.addWidget(self.seqs_container)
        
        seq_widget = QWidget()
        seq_widget.setLayout(seq_section)
        single_layout.addWidget(seq_widget)
        
        bond_section = QVBoxLayout()
        bond_header = QHBoxLayout()
        bond_title = QLabel("Covalent Bonds")
        bond_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #1e293b;")
        bond_header.addWidget(bond_title)
        add_bond_btn = QPushButton("+ Add covalent bond")
        add_bond_btn.setObjectName("OutlineButton")
        add_bond_btn.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        add_bond_btn.clicked.connect(self.add_bond)
        bond_header.addWidget(add_bond_btn)
        bond_link = QLabel("<a href='https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md#covalent_bonds'>Covalent Bonds Tips</a>")
        bond_link.setTextFormat(Qt.TextFormat.RichText)
        bond_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        bond_link.setOpenExternalLinks(True)
        bond_link.setStyleSheet("color: #0f172a; font-size: 11px;")
        bond_header.addSpacing(10)
        bond_header.addWidget(bond_link)
        bond_header.addStretch()
        bond_section.addLayout(bond_header)
        
        self.bonds_container = QWidget()
        self.bonds_layout = QVBoxLayout(self.bonds_container)
        self.bonds_layout.setContentsMargins(0, 0, 0, 0)
        self.bonds_layout.setSpacing(10)
        self.covalent_bonds = []
        
        bond_section.addWidget(self.bonds_container)
        
        bond_widget = QWidget()
        bond_widget.setLayout(bond_section)
        single_layout.addWidget(bond_widget)
        
        single_layout.addStretch()
        single_scroll.setWidget(single_container)
        
        self.prediction_tabs = QTabWidget()
        self.prediction_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.prediction_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                min-width: 140px;
                background-color: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #e0f2fe;
                color: #0369a1;
                border-color: #7dd3fc;
            }
        """)
        self.prediction_tabs.tabBar().setExpanding(True)
        self.prediction_tabs.addTab(single_scroll, "Single Prediction")
        
        self.batch_widget = BatchPredictionWidget()
        self.prediction_tabs.addTab(self.batch_widget, "Batch Prediction")
        
        card_tabs = Card("Prediction Mode")
        card_tabs.layout.setContentsMargins(12, 10, 12, 12)
        card_tabs.layout.setSpacing(8)
        card_tabs.add_widget(self.prediction_tabs)
        layout.addWidget(card_tabs)
        
        card_params = Card("Model Parameter")
        p_grid = QGridLayout()
        p_grid.addWidget(QLabel("<span style='color:red;'>*</span> Sample number"), 0, 0)
        self.inp_sample = QComboBox()
        self.inp_sample.addItems(["1", "5", "10"])
        p_grid.addWidget(self.inp_sample, 1, 0)
        
        p_grid.addWidget(QLabel("<span style='color:red;'>*</span> Recycle numbers"), 0, 1)
        self.inp_recycle = QLineEdit("10")
        p_grid.addWidget(self.inp_recycle, 1, 1)
        
        p_grid.addWidget(QLabel("<span style='color:red;'>*</span> Diffusion steps"), 0, 2)
        self.inp_diffusion = QLineEdit("200")
        p_grid.addWidget(self.inp_diffusion, 1, 2)
        
        p_grid.addWidget(QLabel("Model seeds (optional)"), 0, 3)
        self.inp_seeds = QLineEdit()
        self.inp_seeds.setPlaceholderText("e.g. 123,456")
        p_grid.addWidget(self.inp_seeds, 1, 3)
        
        card_params.add_layout(p_grid)
        card_params.layout.setContentsMargins(16, 12, 16, 12)
        card_params.layout.setSpacing(10)
        layout.addWidget(card_params)
        
        self.log_console = QTextEdit()
        self.log_console.setObjectName("LogConsole")
        self.log_console.setFixedHeight(150)
        self.log_console.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("Protenix execution logs will appear here...")
        layout.addWidget(QLabel("<b>Console Output:</b>"))
        layout.addWidget(self.log_console)

        submit_layout = QHBoxLayout()
        self.generate_json_btn = QPushButton("Preview JSON")
        self.generate_json_btn.setObjectName("OutlineButton")
        self.generate_json_btn.setStyleSheet("font-size: 13px; padding: 8px 24px;")
        self.generate_json_btn.clicked.connect(self.generate_json)
        
        self.submit_btn = QPushButton("Submit & Run Protenix")
        self.submit_btn.setObjectName("PrimaryButton")
        self.submit_btn.clicked.connect(self.run_prediction)
        
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                font-weight: bold;
                padding: 8px 24px;
                border-radius: 20px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #dc2626; }
            QPushButton:disabled { background-color: #fca5a5; }
        """)
        self.abort_btn.clicked.connect(self.abort_prediction)
        self.abort_btn.setEnabled(False)
        
        submit_layout.addStretch()
        submit_layout.addWidget(self.generate_json_btn)
        submit_layout.addSpacing(10)
        submit_layout.addWidget(self.submit_btn)
        submit_layout.addSpacing(10)
        submit_layout.addWidget(self.abort_btn)
        submit_layout.addStretch()
        layout.addLayout(submit_layout)
        
        layout.addStretch()
        scroll_area.setWidget(container)
        return scroll_area
        
    def add_sequence(self):
        seq_number = len(self.sequences) + 1
        seq_widget = SequenceWidget(seq_number, self)
        self.sequences.append(seq_widget)
        self.seqs_layout.addWidget(seq_widget)
        self.renumber_sequences()
        
    def remove_sequence(self, seq_widget, force=False):
        if seq_widget in self.sequences and (len(self.sequences) > 1 or force):
            self.sequences.remove(seq_widget)
            seq_widget.setParent(None)
            self.renumber_sequences()
            
    def renumber_sequences(self):
        for i, seq in enumerate(self.sequences):
            seq.seq_number = i + 1
            seq.findChild(QLabel).setText(f"Sequence {i + 1}")
            
    def add_bond(self):
        bond_number = len(self.covalent_bonds) + 1
        bond_widget = CovalentBondWidget(bond_number, self)
        self.covalent_bonds.append(bond_widget)
        self.bonds_layout.addWidget(bond_widget)
        self.renumber_bonds()
        
    def remove_bond(self, bond_widget):
        if bond_widget in self.covalent_bonds:
            self.covalent_bonds.remove(bond_widget)
            bond_widget.setParent(None)
            self.renumber_bonds()
            
    def renumber_bonds(self):
        for i, bond in enumerate(self.covalent_bonds):
            bond.bond_number = i + 1
            bond.findChild(QLabel).setText(f"Covalent bond {i + 1}")
            
    def reset_all_inputs(self):
        # Reset Global
        self.inp_name.setText("protenix_prediction_job_1")
        self.combo_model.setCurrentIndex(0)
        self.combo_device.setCurrentIndex(0)
        self.inp_cuda_home.setText(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        self.toggle_msa.btn_true.setChecked(True)
        self.toggle_msa.btn_false.setChecked(False)
        self.toggle_template.btn_false.setChecked(True)
        self.toggle_template.btn_true.setChecked(False)
        self.toggle_rna_msa.btn_false.setChecked(True)
        self.toggle_rna_msa.btn_true.setChecked(False)
        self.inp_msa_dir.clear()
        
        # Reset Sequences
        for seq in self.sequences[:]:
            self.remove_sequence(seq, force=True)
        if not self.sequences:
            self.add_sequence()
        
        # Reset Bonds
        for bond in self.covalent_bonds[:]:
            self.remove_bond(bond)
            
        # Reset Settings
        self.inp_seeds.clear()
        self.inp_sample.setCurrentIndex(0)
        self.inp_recycle.setText("10")
        self.inp_diffusion.setText("200")
        
    def load_json_to_ui(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input JSON File",
            "",
            "JSON Files (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("JSON list is empty.")
                job_data = data[0]
            else:
                job_data = data
                
            # Clear current UI first
            self.reset_all_inputs()
            # **IMPORTANT**: Force application to process events so the reset is fully applied
            QApplication.processEvents()
            
            # Clear the default sequence added by reset_all_inputs
            for seq in self.sequences[:]:
                self.remove_sequence(seq, force=True)
            QApplication.processEvents()
            
            # Load Name
            if "name" in job_data:
                self.inp_name.setText(job_data["name"])
                
            # Load Sequences
            if "sequences" in job_data:
                for seq_item in job_data["sequences"]:
                    self.add_sequence()
                    seq_widget = self.sequences[-1]
                    
                    # Determine type and data
                    mol_type = "Protein"
                    entity_data = {}
                    seq_text_value = ""
                    
                    if "proteinChain" in seq_item:
                        mol_type = "Protein"
                        entity_data = seq_item["proteinChain"]
                        seq_text_value = entity_data.get("sequence", "")
                    elif "dnaSequence" in seq_item:
                        mol_type = "DNA"
                        entity_data = seq_item["dnaSequence"]
                        seq_text_value = entity_data.get("sequence", "")
                    elif "rnaSequence" in seq_item:
                        mol_type = "RNA"
                        entity_data = seq_item["rnaSequence"]
                        seq_text_value = entity_data.get("sequence", "")
                    elif "ligand" in seq_item:
                        mol_type = "Ligand"
                        entity_data = seq_item["ligand"]
                        seq_text_value = entity_data.get("ligand", entity_data.get("smiles", ""))
                    elif "ion" in seq_item:
                        mol_type = "Ion"
                        entity_data = seq_item["ion"]
                        seq_text_value = entity_data.get("ion", entity_data.get("name", ""))
                    else:
                        continue
                        
                    idx = seq_widget.mol_combo.findText(mol_type)
                    if idx >= 0:
                        seq_widget.mol_combo.setCurrentIndex(idx)
                        
                    # Set sequence text after changing the combo
                    seq_widget.seq_text.setText(seq_text_value)
                        
                    # Load common properties
                    if "count" in entity_data:
                        seq_widget.inp_copy.setText(str(entity_data["count"]))
                    if "id" in entity_data:
                        seq_widget.inp_id.setText(", ".join(entity_data["id"]))
                        
                    # Load modifications
                    if "modifications" in entity_data:
                        mod_types = []
                        mod_pos = []
                        for mod in entity_data["modifications"]:
                            if "ptmType" in mod:
                                mod_types.append(mod["ptmType"])
                                mod_pos.append(str(mod.get("ptmPosition", "")))
                            elif "modificationType" in mod:
                                mod_types.append(mod["modificationType"])
                                mod_pos.append(str(mod.get("basePosition", "")))
                                
                        if mod_types:
                            seq_widget.inp_mod_type.setText("; ".join(mod_types))
                            seq_widget.inp_mod_pos.setText("; ".join(mod_pos))
                            
                    # Load MSA/Templates
                    if "pairedMsaPath" in entity_data:
                        seq_widget.paired_msa_path.setText(entity_data["pairedMsaPath"])
                    if "unpairedMsaPath" in entity_data:
                        seq_widget.unpaired_msa_path.setText(entity_data["unpairedMsaPath"])
                    if "templatesPath" in entity_data:
                        seq_widget.templates_path.setText(entity_data["templatesPath"])
                        
                    # Load Constraints
                    if "constraint" in entity_data:
                        seq_widget.inp_constraint.setText(json.dumps(entity_data["constraint"]))
                    if "pocket_constraint" in entity_data:
                        seq_widget.inp_pocket_constraint.setText(json.dumps(entity_data["pocket_constraint"]))
                        
            # If no sequences were added, add an empty one
            if not self.sequences:
                self.add_sequence()
                
            # Load Covalent Bonds
            if "covalentBonds" in job_data:
                for bond in job_data["covalentBonds"]:
                    self.add_bond()
                    bond_widget = self.covalent_bonds[-1]
                    # Format: {"entities": [{"id": ["A", "1"], "atom": "C"}, ...]}
                    entities = bond.get("entities", [])
                    if len(entities) >= 1:
                        e1 = entities[0]
                        if "id" in e1 and len(e1["id"]) >= 2:
                            bond_widget.inp_entity1.setText(e1["id"][0])
                            bond_widget.inp_pos1.setText(str(e1["id"][1]))
                        if "atom" in e1:
                            bond_widget.inp_atom1.setText(e1["atom"])
                            
                    if len(entities) >= 2:
                        e2 = entities[1]
                        if "id" in e2 and len(e2["id"]) >= 2:
                            bond_widget.inp_entity2.setText(e2["id"][0])
                            bond_widget.inp_pos2.setText(str(e2["id"][1]))
                        if "atom" in e2:
                            bond_widget.inp_atom2.setText(e2["atom"])
                            
            QMessageBox.information(self, "Success", "JSON loaded successfully.")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load JSON:\n{str(e)}")

    def collect_job_data(self):
        name = self.inp_name.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Name cannot be empty.")
            return None
            
        # Validate sequences
        sequences_data = []
        for seq_widget in self.sequences:
            if not seq_widget.is_valid():
                QMessageBox.warning(self, "Validation Error", f"Please fill in all required fields for Sequence {seq_widget.seq_number}.")
                return None
            seq_error = seq_widget.validate_sequence()
            if seq_error:
                QMessageBox.warning(self, "Validation Error", seq_error)
                return None
            sequences_data.append(seq_widget.get_data())
            
        # Validate covalent bonds
        bonds_data = []
        for bond_widget in self.covalent_bonds:
            if bond_widget.is_valid():
                bonds_data.append(bond_widget.get_data())

        # Process MSA directory if provided
        msa_dir_input = self.inp_msa_dir.text().strip()
        if msa_dir_input and os.path.isdir(msa_dir_input):
            protein_chain_idx = 1
            rna_chain_idx = 1
            
            # Find all relevant files in directory
            all_msa_files = []
            for root, _, files in os.walk(msa_dir_input):
                for file in files:
                    if file.endswith(('.a3m', '.msa', '.hhr')):
                        all_msa_files.append(os.path.abspath(os.path.join(root, file)))
            
            needs_manual_mapping = False
            sequences_to_map = []
            
            for seq_idx, seq in enumerate(sequences_data):
                # Handle Protein Chain MSA
                if "proteinChain" in seq:
                    entity_dict = seq["proteinChain"]
                    
                    # Try to find MSA files for this specific chain index
                    # Possible structures:
                    # 1. msa_dir_input/msa/{idx}/pairing.a3m
                    # 2. msa_dir_input/{idx}/pairing.a3m
                    # 3. msa_dir_input/pairing.a3m (if only 1 chain)
                    
                    paths_to_check = [
                        os.path.join(msa_dir_input, "msa", str(protein_chain_idx)),
                        os.path.join(msa_dir_input, str(protein_chain_idx)),
                        msa_dir_input
                    ]
                    
                    found_any = False
                    for base_path in paths_to_check:
                        if os.path.isdir(base_path):
                            # Check for paired
                            paired_path = os.path.join(base_path, "pairing.a3m")
                            if not os.path.exists(paired_path):
                                paired_path = os.path.join(base_path, "pairing.msa")
                                
                            # Check for unpaired
                            unpaired_path = os.path.join(base_path, "non_pairing.a3m")
                            if not os.path.exists(unpaired_path):
                                unpaired_path = os.path.join(base_path, "non_pairing.msa")
                                
                            # Check for templates
                            templates_path = os.path.join(base_path, "concat.hhr")
                            if not os.path.exists(templates_path):
                                templates_path = os.path.join(base_path, "hmmsearch.a3m")
                            
                            if "pairedMsaPath" not in entity_dict and os.path.exists(paired_path):
                                entity_dict["pairedMsaPath"] = os.path.abspath(paired_path)
                                found_any = True
                            if "unpairedMsaPath" not in entity_dict and os.path.exists(unpaired_path):
                                entity_dict["unpairedMsaPath"] = os.path.abspath(unpaired_path)
                                found_any = True
                            if "templatesPath" not in entity_dict and os.path.exists(templates_path):
                                entity_dict["templatesPath"] = os.path.abspath(templates_path)
                                found_any = True
                                
                            if found_any:
                                break # Found files for this chain, move to next chain
                                
                    # If we didn't find any files through automatic matching but files exist in the dir
                    # AND the user hasn't already manually provided paths in the UI
                    has_manual_paths = any(k in entity_dict for k in ["pairedMsaPath", "unpairedMsaPath", "templatesPath"])
                    if not found_any and all_msa_files and not has_manual_paths:
                        needs_manual_mapping = True
                        
                    sequences_to_map.append({
                        'type': 'Protein',
                        'idx': protein_chain_idx,
                        'global_idx': seq_idx
                    })
                    
                    protein_chain_idx += 1
                    
                # Handle RNA Sequence MSA
                elif "rnaSequence" in seq:
                    entity_dict = seq["rnaSequence"]
                    
                    # Similar path logic for RNA, but typically looking for non_pairing.a3m/msa
                    paths_to_check = [
                        os.path.join(msa_dir_input, "msa", str(rna_chain_idx)),
                        os.path.join(msa_dir_input, str(rna_chain_idx)),
                        msa_dir_input
                    ]
                    
                    found_any = False
                    for base_path in paths_to_check:
                        if os.path.isdir(base_path):
                            # For RNA, check for ANY .a3m or .msa file in the directory
                            local_files = []
                            try:
                                local_files = [f for f in os.listdir(base_path) if f.endswith(('.a3m', '.msa'))]
                            except Exception:
                                pass
                                
                            # ONLY auto-match if exactly 1 file is found
                            if len(local_files) == 1:
                                if "unpairedMsaPath" not in entity_dict:
                                    entity_dict["unpairedMsaPath"] = os.path.abspath(os.path.join(base_path, local_files[0]))
                                    found_any = True
                                break
                                
                    has_manual_paths = "unpairedMsaPath" in entity_dict
                    if not found_any and all_msa_files and not has_manual_paths:
                        needs_manual_mapping = True
                        
                    sequences_to_map.append({
                        'type': 'RNA',
                        'idx': rna_chain_idx,
                        'global_idx': seq_idx
                    })
                                
                    rna_chain_idx += 1
            
            # If automatic matching failed for some chains and we have files, pop up dialog
            if needs_manual_mapping and all_msa_files:
                dialog = MSAMappingDialog(sequences_to_map, all_msa_files, self)
                if dialog.exec() == QDialog.DialogCode.Accepted and dialog.mapping_result:
                    # Apply manual mappings
                    for mapping in dialog.mapping_result:
                        seq_idx = mapping['seq_idx_global']
                        seq_dict = sequences_data[seq_idx]
                        
                        if "proteinChain" in seq_dict:
                            seq_dict["proteinChain"].update(mapping['mapping'])
                        elif "rnaSequence" in seq_dict:
                            seq_dict["rnaSequence"].update(mapping['mapping'])
                else:
                    # User cancelled or closed, return None to abort generation
                    QMessageBox.warning(self, "Cancelled", "MSA mapping cancelled. Job generation aborted.")
                    return None

        # Create job data according to Protenix JSON format
        job_data = {
            "name": name,
            "sequences": sequences_data
        }
        
        if bonds_data:
            job_data["covalent_bonds"] = bonds_data
            
        # Add seed if empty
        seed_val = self.inp_seeds.text().strip()
        if not seed_val:
            seed_val = str(random.randint(1, 999999))
            self.inp_seeds.setText(seed_val)
            
        return job_data
        
    def abort_prediction(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.log_message("[*] Aborting prediction... Please wait.")
            self.worker.stop()
            self.abort_btn.setEnabled(False)

    def generate_json_only(self):
        job_data = self.collect_job_data()
        if not job_data:
            return
            
        try:
            # Save to outputs/<task_name>/input.json to match run path
            task_name = self.inp_name.text().strip()
            if not task_name:
                task_name = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs", task_name))
            os.makedirs(out_dir, exist_ok=True)
            
            file_path = os.path.join(out_dir, "input.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([job_data], f, indent=2)
            
            # Open the file with the default system application
            try:
                if platform.system() == 'Windows':
                    os.startfile(file_path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.Popen(['open', file_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', file_path])
            except Exception as open_e:
                print(f"Failed to open file automatically: {open_e}")
                
            QMessageBox.information(self, "Success", f"JSON file generated and opened:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save JSON file:\n{str(e)}")

    def generate_json(self):
        if hasattr(self, "prediction_tabs") and self.prediction_tabs.currentIndex() == 1:
            if hasattr(self, "batch_widget"):
                self.batch_widget.generate_batch_json()
            return
        self.generate_json_only()
            
    def open_batch_prediction(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Prediction")
        dialog.resize(1200, 700)
        layout = QVBoxLayout(dialog)
        widget = BatchPredictionWidget(dialog, show_close=True, close_callback=dialog.accept)
        layout.addWidget(widget)
        dialog.exec()

    def log_message(self, message):
        self.log_console.append(message)
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    def run_prediction(self):
        script_path = self.inp_script if isinstance(self.inp_script, str) else self.inp_script.text().strip()
        cuda_home = self.inp_cuda_home.text().strip()
        device = self.combo_device.currentText()
        model_name = self.combo_model.currentText()
        
        if not script_path:
            QMessageBox.warning(self, "Validation Error", "Protenix Entry script/command cannot be empty.")
            return
            
        is_batch = hasattr(self, "prediction_tabs") and self.prediction_tabs.currentIndex() == 1
        if is_batch:
            if not hasattr(self, "batch_widget"):
                QMessageBox.warning(self, "Validation Error", "Batch widget not initialized.")
                return
            jobs = self.batch_widget.get_batch_jobs()
            if not jobs:
                return
            job_data = jobs
            batch_name = self.inp_name.text().strip() or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs", batch_name))
        else:
            job_data = self.collect_job_data()
            if not job_data:
                return
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs", job_data['name']))
        
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Running...")
        self.abort_btn.setEnabled(True)
        self.log_console.clear()
        
        if is_batch:
            self.current_task_data = []
            for job in jobs:
                job_name = job.get("name", "batch_job")
                job_dir = os.path.join(out_dir, job_name)
                task_data = {
                    'name': job_name,
                    'dir': job_dir,
                    'status': 'Running',
                    'inputs_json': os.path.join(job_dir, "inputs.json"),
                    'samples': []
                }
                self.prediction_history_tasks.insert(0, task_data)
                self.current_task_data.append(task_data)
        else:
            task_name = out_dir.split(os.sep)[-1]
            self.current_task_data = {
                'name': task_name,
                'dir': out_dir,
                'status': 'Running',
                'inputs_json': os.path.join(out_dir, "inputs.json"),
                'samples': []
            }
            self.prediction_history_tasks.insert(0, self.current_task_data)
        
        self.worker = ProtenixWorker(
            job_data=job_data, 
            out_dir=out_dir, 
            script_path=script_path, 
            cuda_home=cuda_home, 
            device=device, 
            model_name=model_name,
            use_msa=self.toggle_msa.get_value(),
            use_template=self.toggle_template.get_value(),
            use_rna_msa=self.toggle_rna_msa.get_value(),
            seeds=self.inp_seeds.text().strip(),
            sample_num=self.inp_sample.currentText(),
            recycle=self.inp_recycle.text().strip(),
            diffusion_steps=self.inp_diffusion.text().strip(),
            existing_json_path=os.path.join(out_dir, "batch_inputs.json" if is_batch else "input.json")
        )
        self.worker.log_signal.connect(self.log_message)
        self.worker.finished_signal.connect(self.on_prediction_finished)
        self.worker.start()

    def collect_samples_from_dir(self, task_dir):
        samples = []
        seed_dirs = []
        for root, dirs, files in os.walk(task_dir):
            for dir_name in dirs:
                if dir_name.startswith("seed_"):
                    seed_dirs.append(os.path.join(root, dir_name))
        
        for seed_dir in seed_dirs:
            predictions_dir = os.path.join(seed_dir, "predictions")
            if os.path.exists(predictions_dir):
                cif_files = glob.glob(os.path.join(predictions_dir, "*.cif"))
                
                for cif_file in cif_files:
                    basename = os.path.basename(cif_file)
                    sample_num = None
                    
                    match = re.search(r'sample_(\d+)', basename)
                    if match:
                        sample_num = int(match.group(1))
                    
                    json_file = None
                    if sample_num is not None:
                        json_pattern = os.path.join(predictions_dir, f"*sample_{sample_num}*.json")
                        json_files = glob.glob(json_pattern)
                        if json_files:
                            json_file = json_files[0]
                    
                    confidence = ""
                    plddt = None
                    gpde = None
                    ptm = None
                    iptm = None
                    if json_file and os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                ptm = data.get("ptm") or data.get("pTM")
                                iptm = data.get("iptm") or data.get("ipTM")
                                if ptm is not None and iptm is not None:
                                    confidence = f"pTM = {ptm:.2f}\nipTM = {iptm:.2f}"
                                if 'plddt' in data:
                                    if isinstance(data['plddt'], list):
                                        plddt = data['plddt']
                                    else:
                                        plddt = [data['plddt']]
                                elif 'chain_plddt' in data:
                                    if isinstance(data['chain_plddt'], list):
                                        plddt = data['chain_plddt']
                                    else:
                                        plddt = [data['chain_plddt']]
                                if 'gpde' in data:
                                    gpde = data['gpde']
                                elif 'ranking_confidence' in data:
                                    gpde = data['ranking_confidence']
                                elif 'ranking_score' in data:
                                    gpde = data['ranking_score']
                        except Exception:
                            pass
                    
                    samples.append({
                        'name': basename,
                        'sample_num': sample_num,
                        'cif_file': cif_file,
                        'json_file': json_file,
                        'confidence': confidence,
                        'plddt': plddt,
                        'gpde': gpde,
                        'ptm': ptm,
                        'iptm': iptm
                    })
        
        samples.sort(key=lambda x: x['sample_num'] if x['sample_num'] is not None else 9999)
        return samples
        
    def on_prediction_finished(self, success, message):
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("Submit & Run Protenix")
        self.abort_btn.setEnabled(False)
        
        if success:
            self.log_message(f"\n[SUCCESS] {message}")
            if isinstance(self.current_task_data, list):
                for task_data in self.current_task_data:
                    task_dir = task_data['dir']
                    samples = self.collect_samples_from_dir(task_dir)
                    if not samples and not os.path.exists(task_dir):
                        samples = self.collect_samples_from_dir(os.path.dirname(task_dir))
                    err_dir = os.path.join(task_dir, "ERR")
                    has_err = os.path.exists(err_dir)
                    task_data['samples'] = samples
                    task_data['has_err'] = has_err
                    task_data['err_dir'] = err_dir
                    task_data['status'] = 'Finished' if samples else ('Failed' if has_err else 'No Results')
            else:
                self.current_task_data['status'] = 'Finished'
                out_dir = self.current_task_data['dir']
                samples = self.collect_samples_from_dir(out_dir)
                self.current_task_data['samples'] = samples
            
            QMessageBox.information(self, "Job Completed", message)
        else:
            self.log_message(f"\n[ERROR] {message}")
            if isinstance(self.current_task_data, list):
                for task_data in self.current_task_data:
                    task_data['status'] = 'Failed'
            else:
                self.current_task_data['status'] = 'Failed'
            QMessageBox.critical(self, "Job Failed", message)

    # --- 渲染表格 ---
    def refresh_table(self, table_widget, data):
        table_widget.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            # 仅渲染表格头指定的列数，避免把隐藏的绝对路径渲染出来
            for col_idx in range(table_widget.columnCount()):
                col_data = row_data[col_idx]
                
                # 针对不同列渲染交互按钮
                if col_idx == 2 and col_data == "view": # Input (for Predictions)
                    btn = QPushButton("view")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.view_input(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                elif col_idx == 4 and col_data == "Open": # Output (for Predictions)
                    btn = QPushButton("Open")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.open_output(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                elif col_idx == 5 and col_data == "view": # Detail (for Predictions)
                    btn = QPushButton("view")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.view_structure(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                elif col_idx == 4 and col_data == "view": # Input (for Designs)
                    btn = QPushButton("view")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.view_input(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                elif col_idx == 6 and col_data == "Open": # Output (for Designs)
                    btn = QPushButton("Open")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.open_output(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                elif col_idx == 7 and col_data == "view": # Detail (for Designs)
                    btn = QPushButton("view")
                    btn.setStyleSheet("color: #3b82f6; border: none; background: transparent; text-decoration: underline;")
                    btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    btn.clicked.connect(lambda checked, r=row_idx: self.view_structure(r))
                    table_widget.setCellWidget(row_idx, col_idx, btn)
                    
                else: # 普通文本渲染
                    item = QTableWidgetItem(str(col_data))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if col_idx == 1: # Status (for Predictions)
                        if col_data == "Finished": item.setForeground(QColor("#16a34a"))
                        elif col_data == "Running": item.setForeground(QColor("#eab308"))
                        elif col_data == "Failed": item.setForeground(QColor("#ef4444"))
                    elif col_idx == 2: # Status (for Designs)
                        if col_data == "Finished": item.setForeground(QColor("#16a34a"))
                        elif col_data == "Running": item.setForeground(QColor("#eab308"))
                        elif col_data == "Failed": item.setForeground(QColor("#ef4444"))
                    table_widget.setItem(row_idx, col_idx, item)

    # --- 各种交互按钮的功能实现 ---
    def view_input(self, row_idx):
        # 检查是否是Prediction History页面的数据
        if len(self.prediction_history_data) > row_idx:
            task_name = self.prediction_history_data[row_idx][0]
            out_dir = self.prediction_history_data[row_idx][6] if len(self.prediction_history_data[row_idx]) > 6 else None
        else:
            # 否则是Designs页面的数据
            task_name = self.design_history_data[row_idx][0]
            out_dir = self.design_history_data[row_idx][8] if len(self.design_history_data[row_idx]) > 8 else None
        
        if not out_dir or not os.path.exists(os.path.join(out_dir, "inputs.json")):
            QMessageBox.warning(self, "Not Found", f"inputs.json not found")
            return
            
        with open(os.path.join(out_dir, "inputs.json"), 'r', encoding='utf-8') as f:
            content = f.read()
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Input JSON - {task_name}")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        text_edit.setText(content)
        layout.addWidget(text_edit)
        dialog.exec()

    def open_output(self, row_idx):
        # 检查是否是Prediction History页面的数据
        if len(self.prediction_history_data) > row_idx:
            out_dir = self.prediction_history_data[row_idx][6] if len(self.prediction_history_data[row_idx]) > 6 else None
        else:
            # 否则是Designs页面的数据
            out_dir = self.design_history_data[row_idx][8] if len(self.design_history_data[row_idx]) > 8 else None
        
        if not out_dir or not os.path.exists(out_dir):
            QMessageBox.warning(self, "Not Found", f"Output directory not found")
            return
            
        if sys.platform == 'win32':
            os.startfile(out_dir)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', out_dir])
        else:
            subprocess.Popen(['xdg-open', out_dir])

    def view_structure(self, row_idx):
        # 检查是否是Prediction History页面的数据
        if len(self.prediction_history_data) > row_idx:
            out_dir = self.prediction_history_data[row_idx][6] if len(self.prediction_history_data[row_idx]) > 6 else None
            task_name = self.prediction_history_data[row_idx][0]
        else:
            # 否则是Designs页面的数据
            out_dir = self.design_history_data[row_idx][8] if len(self.design_history_data[row_idx]) > 8 else None
            task_name = self.design_history_data[row_idx][0]
        
        if not out_dir:
            QMessageBox.warning(self, "Not Found", f"Output directory not found")
            return
            
        # 递归寻找所有的.cif文件，解决多层目录问题
        cif_files = glob.glob(os.path.join(out_dir, "**", "*.cif"), recursive=True)
        
        if not cif_files:
            QMessageBox.warning(self, "Not Found", f"No .cif file found in:\n{out_dir}")
            return
        
        if len(cif_files) == 1:
            cif_file = cif_files[0]
            self.open_in_pymol(cif_file)
        else:
            # 如果有多个cif文件，让用户选择
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Select Structure - {task_name}")
            dialog.resize(600, 400)
            layout = QVBoxLayout(dialog)
            
            label = QLabel("Multiple structure files found. Please select one:")
            layout.addWidget(label)
            
            from PyQt6.QtWidgets import QListWidget
            list_widget = QListWidget()
            for cf in cif_files:
                list_widget.addItem(os.path.basename(cf))
            list_widget.setCurrentRow(0)
            layout.addWidget(list_widget)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            ok_btn = QPushButton("Open in PyMOL")
            ok_btn.setObjectName("PrimaryButton")
            ok_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(ok_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_index = list_widget.currentRow()
                cif_file = cif_files[selected_index]
                self.open_in_pymol(cif_file)
                
    def open_structure_viewer(self, cif_file, title_text, plddt_data=None):
        #region debug-point
        _dbg_log("preview_open_structure_viewer", {
            "web_engine": WEB_ENGINE_AVAILABLE,
            "cif_file": cif_file,
            "plddt_type": str(type(plddt_data)),
            "plddt_len": len(plddt_data) if isinstance(plddt_data, list) else None
        })
        #endregion debug-point
        if not WEB_ENGINE_AVAILABLE:
            QMessageBox.warning(self, "Viewer Unavailable", "3D viewer requires PyQt6-WebEngine. Falling back to PyMOL.")
            self.open_in_pymol(cif_file)
            return
        
        try:
            with open(cif_file, 'r', encoding='utf-8') as f:
                cif_content = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read structure file:\n{str(e)}")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Structure Viewer - {title_text}")
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        
        viewer = QWebEngineView()
        cif_js = json.dumps(cif_content)
        
        chart_html = ""
        chart_script = ""
        if plddt_data is not None:
            # 如果是单值，为了防止报错也转成数组处理
            if not isinstance(plddt_data, list):
                plddt_data = [plddt_data]
            plddt_js = json.dumps(plddt_data)
            chart_html = """
            <div id="chart-container" style="position: absolute; top: 20px; left: 20px; width: 400px; height: 250px; background: rgba(255,255,255,0.9); border-radius: 6px; border: 1px solid #e2e8f0; padding: 10px; z-index: 1000;">
                <canvas id="plddt-chart"></canvas>
            </div>
            """
            chart_script = f"""
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                // 使用轮询等待Chart.js加载完成并找到canvas
                function initChart() {{
                    if (typeof Chart === 'undefined') {{
                        setTimeout(initChart, 100);
                        return;
                    }}
                    
                    const canvas = document.getElementById('plddt-chart');
                    if (!canvas) {{
                        setTimeout(initChart, 100);
                        return;
                    }}
                    
                    const ctx = canvas.getContext('2d');
                    const plddtData = {plddt_js};
                    const labels = Array.from({{length: plddtData.length}}, (_, i) => i + 1);
                    
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: labels,
                            datasets: [{{
                                label: 'pLDDT',
                                data: plddtData,
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: true
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }},
                                title: {{ display: true, text: 'Per-residue pLDDT', font: {{ size: 14 }} }}
                            }},
                            scales: {{
                                y: {{ min: 0, max: 100, ticks: {{ font: {{ size: 10 }} }} }},
                                x: {{ ticks: {{ font: {{ size: 10 }}, maxTicksLimit: 10 }} }}
                            }}
                        }}
                    }});
                }}
                
                // 启动轮询
                setTimeout(initChart, 100);
            </script>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <style>
                html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
                #viewer {{ width: 100%; height: 100%; }}
                .legend {{
                    position: absolute;
                    bottom: 20px;
                    right: 20px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border-radius: 6px;
                    border: 1px solid #e2e8f0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    font-size: 12px;
                    z-index: 1000;
                }}
                .legend-title {{ font-weight: bold; margin-bottom: 8px; color: #1e293b; }}
                .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
                .color-box {{ width: 16px; height: 16px; margin-right: 8px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div id="viewer"></div>
            {chart_html}
            <div class="legend">
                <div class="legend-title">pLDDT (Confidence)</div>
                <div class="legend-item"><div class="color-box" style="background: #0053d6;"></div>Very High (>90)</div>
                <div class="legend-item"><div class="color-box" style="background: #65cbf3;"></div>High (70-90)</div>
                <div class="legend-item"><div class="color-box" style="background: #ffdb13;"></div>Low (50-70)</div>
                <div class="legend-item"><div class="color-box" style="background: #ff7d45;"></div>Very Low (<50)</div>
            </div>
            <script>
                const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
                const cif = {cif_js};
                viewer.addModel(cif, "cif");
                
                // Set color based on b-factor (pLDDT is stored in b-factor column)
                viewer.setStyle({{}}, {{ cartoon: {{ 
                    colorscheme: {{
                        prop: 'b',
                        gradient: 'sinebow',
                        min: 50,
                        max: 90,
                        mid: 70
                    }} 
                }} }});
                
                // Using AlphaFold-like color mapping
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorfunc: function(atom) {{
                            if (atom.b > 90) return '#0053d6';
                            if (atom.b > 70) return '#65cbf3';
                            if (atom.b > 50) return '#ffdb13';
                            return '#ff7d45';
                        }}
                    }}
                }});
                
                viewer.zoomTo();
                viewer.render();
            </script>
            {chart_script}
        </body>
        </html>
        """
        #region debug-point
        _dbg_log("preview_html_ready", {"html_len": len(html)})
        #endregion debug-point
        viewer.setHtml(html)
        #region debug-point
        viewer.loadFinished.connect(lambda ok: _dbg_log("preview_load_finished", {"ok": ok}))
        #endregion debug-point
        layout.addWidget(viewer)
        dialog.exec()
                
    def open_in_pymol(self, cif_file):
        # 尝试不同的PyMOL路径
        pymol_paths = ['pymol', '/opt/homebrew/bin/pymol']
        opened = False
        
        for pymol_exec in pymol_paths:
            try:
                subprocess.Popen([pymol_exec, cif_file])
                opened = True
                break
            except Exception as e:
                print(f"Failed to open with {pymol_exec}: {e}")
        
        if not opened:
            QMessageBox.warning(self, "Error", "Could not open PyMOL. Please make sure PyMOL is installed and in your PATH.")

    def create_history_page(self, title_text, headers, data):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        
        header_layout = QHBoxLayout()
        title = QLabel(title_text)
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header_layout.addWidget(title)
        
        # 仅在结构预测页面增加导入按钮
        if title_text == "Predictions":
            header_layout.addStretch()
            
            # 添加信息按钮
            info_btn = QPushButton("i")
            info_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #64748b;
                    border: none;
                    font-size: 16px;
                    font-weight: bold;
                    width: 24px;
                    height: 24px;
                }
                QPushButton:hover {
                    color: #3b82f6;
                }
            """)
            info_btn.set_info_text("""
pLDDT (per-residue confidence): 
- Measures the confidence of each residue's position
- 0-100 scale, higher is better
- >90: Very high quality
- 70-90: Good quality
- <70: Low quality

GPDE (Global Distance Estimate):
- Measures global structure quality
- Higher values indicate better confidence
- Used for ranking multiple predictions

pTM (predicted template modeling):
- Measures overall structure quality
- 0-1 scale, higher is better
- >0.7: Good quality

ipTM (interface pTM):
- Measures interface quality for multi-chain complexes
- 0-1 scale, higher is better
- >0.7: Good interface
            """)
            header_layout.addWidget(info_btn)
            
            load_btn = QPushButton("📂 Load Result Directory")
            load_btn.setObjectName("OutlineButton")
            load_btn.clicked.connect(self.load_history_from_dir)
            header_layout.addWidget(load_btn)
            
        layout.addLayout(header_layout)
        
        table = QTableWidget()
        table.setObjectName(f"Table_{title_text}")
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setStyleSheet("alternate-background-color: #f8fafc;")
        
        self.refresh_table(table, data)
        layout.addWidget(table)
        
        return container
        

            
    def create_expandable_history_page(self, title_text):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        
        header_layout = QHBoxLayout()
        title = QLabel(title_text)
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 添加信息按钮
        info_btn = InfoButton()
        info_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
                border-radius: 12px;
            }
        """)
        info_btn.set_info_text("""
pLDDT (per-residue confidence): 
- Measures the confidence of each residue's position
- 0-100 scale, higher is better
- >90: Very high quality
- 70-90: Good quality
- <70: Low quality

GPDE (Global Distance Estimate):
- Measures global structure quality
- Higher values indicate better confidence
- Used for ranking multiple predictions

pTM (predicted template modeling):
- Measures overall structure quality
- 0-1 scale, higher is better
- >0.7: Good quality

ipTM (interface pTM):
- Measures interface quality for multi-chain complexes
- 0-1 scale, higher is better
- >0.7: Good interface
        """)
        header_layout.addWidget(info_btn)
        
        load_btn = QPushButton("📂 Load Result Directory")
        load_btn.setObjectName("OutlineButton")
        load_btn.clicked.connect(self.load_history_from_dir)
        header_layout.addWidget(load_btn)
        
        self.batch_delete_btn = QPushButton("🗑️ Batch Delete")
        self.batch_delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #fef2f2;
                color: #ef4444;
                border: 1px solid #fecaca;
                border-radius: 15px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #fee2e2;
            }
        """)
        self.batch_delete_btn.clicked.connect(self.batch_delete_tasks)
        header_layout.addWidget(self.batch_delete_btn)
            
        layout.addLayout(header_layout)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setSpacing(6)
        self.tasks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area.setWidget(self.tasks_container)
        layout.addWidget(scroll_area)
        
        return container
        
    def refresh_expandable_history(self):
        # 清除现有内容
        for i in reversed(range(self.tasks_layout.count())):
            self.tasks_layout.itemAt(i).widget().setParent(None)
            
        # 添加任务
        for task in self.prediction_history_tasks:
            task_widget = ExpandableTaskWidget(task)
            self.tasks_layout.addWidget(task_widget)
            
    def load_history_from_dir(self, directory=None, show_msg=True):
        if directory:
            selected_dir = directory
        else:
            selected_dir = QFileDialog.getExistingDirectory(
                self, 
                "Select Main Outputs Directory (e.g. 'outputs')",
                options=QFileDialog.Option.DontUseNativeDialog
            )
        
        if not selected_dir:
            return
            
        loaded_count = 0
        
        # 遍历主目录下的所有子目录
        for item in os.listdir(selected_dir):
            task_dir = os.path.join(selected_dir, item)
            if not os.path.isdir(task_dir):
                continue
                
            task_name = item
            
            # 去重
            if any(t['name'] == task_name for t in self.prediction_history_tasks):
                continue
                
            # 查找inputs.json
            inputs_json_path = os.path.join(task_dir, "inputs.json")
            
            # 查找所有seed目录
            seed_dirs = []
            for root, dirs, files in os.walk(task_dir):
                for dir_name in dirs:
                    if dir_name.startswith("seed_"):
                        seed_dirs.append(os.path.join(root, dir_name))
            
            # 收集所有样本
            samples = []
            for seed_dir in seed_dirs:
                predictions_dir = os.path.join(seed_dir, "predictions")
                if os.path.exists(predictions_dir):
                    # 查找所有cif和对应的json文件
                    cif_files = glob.glob(os.path.join(predictions_dir, "*.cif"))
                    
                    for cif_file in cif_files:
                        # 提取sample编号
                        basename = os.path.basename(cif_file)
                        sample_num = None
                        
                        # 从文件名中提取sample编号
                        import re
                        match = re.search(r'sample_(\d+)', basename)
                        if match:
                            sample_num = int(match.group(1))
                        
                        # 查找对应的json文件
                        json_file = None
                        if sample_num is not None:
                            json_pattern = os.path.join(predictions_dir, f"*sample_{sample_num}*.json")
                            json_files = glob.glob(json_pattern)
                            if json_files:
                                json_file = json_files[0]
                        
                        # 解析confidence和其他参数
                        confidence = ""
                        plddt = None
                        gpde = None
                        ptm = None
                        iptm = None
                        if json_file and os.path.exists(json_file):
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    ptm = data.get("ptm") or data.get("pTM")
                                    iptm = data.get("iptm") or data.get("ipTM")
                                    if ptm is not None and iptm is not None:
                                        confidence = f"pTM = {ptm:.2f}\nipTM = {iptm:.2f}"
                                    # 读取pLDDT - 针对单值或数组
                                    if 'plddt' in data:
                                        if isinstance(data['plddt'], list):
                                            plddt = data['plddt']
                                        else:
                                            plddt = [data['plddt']]  # 转为列表以防图表渲染失败
                                    elif 'chain_plddt' in data:
                                        if isinstance(data['chain_plddt'], list):
                                            plddt = data['chain_plddt']
                                        else:
                                            plddt = [data['chain_plddt']]
                                    # 读取GPDE或其他参数
                                    if 'gpde' in data:
                                        gpde = data['gpde']
                                    elif 'ranking_confidence' in data:
                                        gpde = data['ranking_confidence']
                                    elif 'ranking_score' in data:
                                        gpde = data['ranking_score']
                            except Exception as e:
                                print(f"Failed to parse confidence: {e}")
                        
                        samples.append({
                            'name': basename,
                            'sample_num': sample_num,
                            'cif_file': cif_file,
                            'json_file': json_file,
                            'confidence': confidence,
                            'plddt': plddt,
                            'gpde': gpde,
                            'ptm': ptm,
                            'iptm': iptm
                        })
            
            # 按sample_num排序
            samples.sort(key=lambda x: x['sample_num'] if x['sample_num'] is not None else 9999)
            
            # 检查是否有ERR文件夹
            err_dir = os.path.join(task_dir, "ERR")
            has_err = os.path.exists(err_dir)
            
            # 创建任务数据
            task_data = {
                'name': task_name,
                'dir': task_dir,
                'status': 'Finished' if samples else ('Failed' if has_err else 'No Results'),
                'has_err': has_err,
                'err_dir': err_dir,
                'inputs_json': inputs_json_path,
                'samples': samples
            }
            
            self.prediction_history_tasks.insert(0, task_data)
            loaded_count += 1
            
        if loaded_count > 0:
            self.refresh_expandable_history()
            if show_msg:
                QMessageBox.information(self, "Success", f"Successfully loaded {loaded_count} tasks from:\n{selected_dir}")
        else:
            if show_msg:
                QMessageBox.information(self, "Info", "No new valid task directories found or they are already loaded.")

    def batch_delete_tasks(self):
        selected_tasks = []
        selected_samples = []
        
        for i in range(self.tasks_layout.count()):
            widget = self.tasks_layout.itemAt(i).widget()
            if isinstance(widget, ExpandableTaskWidget):
                if widget.checkbox.isChecked():
                    selected_tasks.append(widget)
                else:
                    # 如果Task未选中，检查其内部是否有选中的Sample
                    for j in range(widget.expanded_layout.count()):
                        sample_frame = widget.expanded_layout.itemAt(j).widget()
                        if hasattr(sample_frame, 'checkbox') and sample_frame.checkbox.isChecked():
                            selected_samples.append((widget, sample_frame.sample_data, sample_frame))
        
        if not selected_tasks and not selected_samples:
            QMessageBox.information(self, "Info", "No tasks or samples selected for batch deletion.")
            return

        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Batch Deletion")
        dialog.resize(350, 150)
        
        layout = QVBoxLayout(dialog)
        
        msg = "Are you sure you want to delete:\n"
        if selected_tasks:
            msg += f"- {len(selected_tasks)} tasks\n"
        if selected_samples:
            msg += f"- {len(selected_samples)} individual samples\n"
            
        msg_label = QLabel(msg)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        delete_files_cb = QCheckBox("Move files to Trash")
        delete_files_cb.setStyleSheet("margin-top: 10px; margin-bottom: 10px; color: #ef4444;")
        layout.addWidget(delete_files_cb)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("Delete Selected")
        confirm_btn.setStyleSheet("background-color: #ef4444; color: white; border: none; padding: 4px 12px; border-radius: 4px;")
        confirm_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(confirm_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                # 1. 批量删除完整任务
                for widget in selected_tasks:
                    task_data = widget.task_data
                    if delete_files_cb.isChecked() and os.path.exists(task_data['dir']):
                        safe_trash_delete(task_data['dir'])
                    
                    if task_data in self.prediction_history_tasks:
                        self.prediction_history_tasks.remove(task_data)
                        
                # 2. 批量删除独立样本
                for task_widget, sample, sample_frame in selected_samples:
                    if delete_files_cb.isChecked():
                        if sample['cif_file'] and os.path.exists(sample['cif_file']):
                            safe_trash_delete(sample['cif_file'])
                        if sample['json_file'] and os.path.exists(sample['json_file']):
                            safe_trash_delete(sample['json_file'])
                    
                    # 从数据中移除并销毁UI
                    if sample in task_widget.task_data['samples']:
                        task_widget.task_data['samples'].remove(sample)
                    sample_frame.setParent(None)
                    sample_frame.deleteLater()
                    
                    # 更新状态（如果删空了）
                    if not task_widget.task_data['samples']:
                        task_widget.task_data['status'] = 'Empty'
                        header_layout = task_widget.main_layout.itemAt(0).layout()
                        # 找到状态标签并更新（跳过复选框, 展开按钮, 标题）
                        for k in range(header_layout.count()):
                            item = header_layout.itemAt(k)
                            if item and item.widget() and isinstance(item.widget(), QLabel):
                                text = item.widget().text()
                                if text in ['Finished', 'Failed', 'No Results', 'Empty']:
                                    item.widget().setText('Empty')
                                    item.widget().setStyleSheet("color: #64748b; font-weight: 600; font-size: 11px;")
                                    break
                
                self.refresh_expandable_history()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete items:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ProtenixServerApp()
    window.show()
    sys.exit(app.exec())
