"""Build the Colab notebook with its reviewed helper code embedded."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    support = (ROOT / "tools" / "colab_support.py").read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("""# MambaCenter：Colab 断点续训与对照实验

默认任务为 **R1_RESUME**：从已验证的第8轮继续到第15轮。先选择 T4 GPU，再依次运行。

- 需要挂载你自己的 Google Drive；任务输出直接保存在 `MyDrive/MambaCenter_Checkpoints/`，每轮写入检查点和日志。挂载失败会停止，不回退到临时磁盘。
- 输入包位于 `MyDrive/MambaCenter_Colab_Inputs/`。支持完整ZIP或4个分块和manifest；会校验完整包及原始第8轮权重的SHA256。
- 训练代码固定为 `d6c1f07dc35e85e5f077b90f7f9b8c7e9a8d62d6`，准备代码内嵌在本Notebook中。数据标注指纹与已保存Kaggle R1对账。
- R1剩7轮按旧T4记录约需9–10小时量级，R2全部15轮约20小时量级，另加准备与评测。下载、编译和免费GPU时长均没有保证。
- 换平台或中断恢复不等于逐步完全复现：历史检查点未保存完整RNG/worker状态。结果标记为探索性，不宣称严格成对实验。
- 断开可能损失当前未完成的轮次。若检查点与history不一致，Notebook停止并保留文件，需补回该轮评价后再继续。
- best按AP50@500选择，最终轮另列；COCO-style评价不等于官方VisDrone忽略区域评价。
"""),
        code("""# 1. 选择任务；此单元不会启动训练
from pathlib import Path
TASK = "R1_RESUME"  # @param ["R1_RESUME", "R2_FRESH", "R0_BASELINE", "R3_CPMAMBA"]
INPUT_DIR = Path("/content/drive/MyDrive/MambaCenter_Colab_Inputs")
DRIVE_ROOT = Path("/content/drive/MyDrive/MambaCenter_Checkpoints")
WORKDIR = Path("/content/MambaCenter_src_d6c1f07")
TARGET_EPOCHS = 15
"""),
        code("""# 2. Google Drive：必须成功挂载；不使用本机GPU
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Google Drive未成功挂载，停止准备。")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
probe = DRIVE_ROOT / ".mambacenter_write_probe"
probe.write_text("ok", encoding="utf-8")
probe.unlink()
print("训练检查点将直接保存在：", DRIVE_ROOT)
"""),
        code(support),
        code("""# 4. 检查实际云端GPU，建立固定版本代码目录
import platform
import torch
if not torch.cuda.is_available():
    raise RuntimeError("当前是CPU运行时。请在代码执行程序→更改运行时类型中选择T4 GPU。")
gpu = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if "T4" not in gpu or capability != (7, 5):
    raise RuntimeError(f"本协议预期T4，实际为{gpu} / {capability}。请先确认硬件，不自动更换实验协议。")
if torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
    raise RuntimeError("当前GPU显存不足。")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"
prepare_repo(WORKDIR)
checked([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
environment = {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
               "python": platform.python_version(), "platform": platform.platform()}
print(environment)
"""),
        code("""# 5. 编译并核验原生扫描后端；失败直接停止
checked([sys.executable, "runtime/selective-scan/build.py", "build_ext",
         "--build-lib", "runtime/selective-scan/packages", "--build-temp", "runtime/selective-scan/build"], cwd=WORKDIR)
paths = [WORKDIR / "runtime/selective-scan/packages", WORKDIR / "runtime/selective-scan/mamba", WORKDIR]
os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + os.environ.get("PYTHONPATH", "")
for path in reversed(paths):
    sys.path.insert(0, str(path))
import selective_scan_cuda
from mambacenter import scan
if scan._cuda_scan is None:
    raise RuntimeError(f"原生Mamba扫描不可用，禁止自动回退参考实现：{scan.CUDA_IMPORT_ERROR}")
print("原生扩展：", selective_scan_cuda.__file__)
"""),
        code("""# 6. 使用与历史运行一致的预训练文件，校验完整SHA256
pretrained = ensure_pretrained(INPUT_DIR)
print(pretrained)
"""),
        code("""# 7. 下载VisDrone并转换标注；核对图片/框数量和标注语义指纹
data_fingerprints = prepare_data(WORKDIR)
print(json.dumps(data_fingerprints, indent=2))
"""),
        code("""# 8. 仅恢复对应任务的检查点；支持本任务后续第9–14轮再次续训
CONFIG_FILE, RUN_NAME, EXPECTED_ARCH = TASKS[TASK]
OUT_DIR = DRIVE_ROOT / RUN_NAME
manifest_dir = DRIVE_ROOT / "session_manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)
resume_state = prepare_run(OUT_DIR, TASK, INPUT_DIR, Path("/content/r1_epoch8_checkpoint.zip"))
environment.update({"data": data_fingerprints, "pretrained": pretrained})
print("任务：", TASK, "输出：", OUT_DIR, "检查点：", resume_state)
"""),
        code("""# 9. 云端数值预检：原生扫描、混合器、DCN与真实数据目标
preflight_path = manifest_dir / f"{RUN_NAME}_preflight_{time.time_ns()}.json"
checked([sys.executable, "tests/gpu_checks.py", "--out", preflight_path], cwd=WORKDIR)
environment["preflight_file"] = str(preflight_path)
print("预检通过。下一单元才会开始正式训练。")
"""),
        code("""# 10. 开始云端训练，逐轮直接写Google Drive；已到15轮则不重复运行
session_path = manifest_dir / f"{RUN_NAME}_{time.time_ns()}.json"
launch_training(WORKDIR, OUT_DIR, TASK, environment, session_path)
"""),
        code("""# 11. 查看真实已保存指标；缺失值显示NA，不用0冒充结果
records = history_records(OUT_DIR / "history.jsonl")
def score(record, key):
    value = record.get("val", {}).get(key)
    return "NA" if value is None else f"{100 * value:.3f}%"
print("epoch | AP@100 | AP@500 | AP50@500 | AP75@500")
for record in records:
    print(record["epoch"], *(score(record, key) for key in ("AP@100", "AP@500", "AP50@500", "AP75@500")), sep=" | ")
print("产物目录：", OUT_DIR)
"""),
    ]
    nb.metadata["mambacenter"] = {"support_sha256": hashlib.sha256(support.encode()).hexdigest(),
                                   "training_source_revision": "d6c1f07dc35e85e5f077b90f7f9b8c7e9a8d62d6"}
    nbf.validate(nb)
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/MambaCenter_Colab_Training.ipynb")
    parser.add_argument("--extra-output", type=Path)
    args = parser.parse_args()
    notebook = build()
    for path in [args.output, args.extra_output]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            nbf.write(notebook, path)
            print(path)
