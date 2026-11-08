# MambaCenter: Kaggle GPU 训练与复现全流程指南

本文档指导如何在 **Kaggle** 的 GPU（Tesla T4 16GB / P100 / A100）上顺利编译、训练与评测 **MambaCenter**（包含 CenterNet 基线、CP-Mamba B5'、IA-Mamba A4' 以及 Addon-VSS）。

---

## 一、为什么需要 Kaggle 专属适配？（解决核心痛点）

原版实验方案运行于私有集群的 **NVIDIA RTX 5090 (32GB VRAM, sm_120)** 上，直接搬到 Kaggle 会遇到以下硬性约束：

| 约束项 | RTX 5090 原版实验 | Kaggle Tesla T4 (主流免费环境) | 我们的适配方案 |
|---|---|---|---|
| **显存大小** | 32 GB（`batch_size: 20` 峰值达 27.8 GB） | 15.9 GB（直接跑必报 CUDA OOM 爆显存） | 专门制定 `kaggle_addon_cp.yaml` 等配置：`batch_size: 8` + `accum_steps: 2`（等效 batch 16），`micro_batch: 4`，显存压至 8–10 GB |
| **混合精度** | 原生 `bf16`（无需 loss scaler） | **T4 硬件不支持原生 bf16**（强开会报错或速度慢 5-10 倍） | `tools/train.py` 升级：自动硬件探测，在 T4/P100 上无缝切换为 **FP16 + PyTorch GradScaler**，发挥 T4 Tensor Core 最大硬件算力 |
| **CUDA 编译** | 针对 sm_120 构建 | 需要针对 Turing sm_75 构建 | 设置 `TORCH_CUDA_ARCH_LIST="7.5"`，使用 `ninja` 在 2 分钟内编译完成 |
| **磁盘配额** | 集群共享大磁盘 | Kaggle `/kaggle/working` 仅 20 GB 空间 | 零拷贝软链接（Symlink），直接读取只读挂载的 VisDrone 原始图片，不复制数据集，仅生成 20MB 的 COCO json |
| **网络下载** | 计算节点离线预下载 | Kaggle 可访问公网，但官方权重源 `dl.yf.io` 偶发超时 | 脚本预下载 GitHub Release 镜像源的 `dla34-ba72cf86.pth` 到 torch hub cache |
| **会话超时** | Slurm 长作业 | 单次交互最多 9 小时，后台作业最多 12 小时 | 支持完整的 `--resume` 断点续训，模型与日志持续写入 `/kaggle/working/runs` |

---

## 二、交付文件清单

在你的本地目录 `D:\cv\` 中已准备好全部资源：

1. **Jupyter Notebook 一键脚本**：
   - 路径：[MambaCenter_Kaggle_Training.ipynb](file:///D:/cv/MambaCenter-cpia/notebooks/MambaCenter_Kaggle_Training.ipynb)（在根目录也有一份镜像副本）
   - 直接导入 Kaggle Notebook 即可一键顺序运行。
2. **Kaggle 专属配置文件**：
   - [configs/kaggle_t4.yaml](file:///D:/cv/MambaCenter-cpia/configs/kaggle_t4.yaml)：CenterNet DLA-34 基准。
   - [configs/kaggle_addon_cp.yaml](file:///D:/cv/MambaCenter-cpia/configs/kaggle_addon_cp.yaml)：CP-Mamba B5' 外挂模型。
   - [configs/kaggle_addon_ia.yaml](file:///D:/cv/MambaCenter-cpia/configs/kaggle_addon_ia.yaml)：IA-Mamba A4' 外挂模型。
3. **打包好的轻量级上传包**：
   - 路径：`D:\cv\MambaCenter_Kaggle_Ready.zip`（仅 10.68 MB，去除了 git/build/log 等冗余）。
4. **升级后的训练脚本**：
   - [tools/train.py](file:///D:/cv/MambaCenter-cpia/tools/train.py)：完美兼容 FP16 + GradScaler、梯度累加、Windows/Linux 跨平台，25 项 CPU 单元测试 100% 通过。

---

## 三、Kaggle 运行实操步骤（手把手）

### 步骤 1：新建 Kaggle Notebook 与环境配置
1. 打开 [Kaggle](https://www.kaggle.com/) -> 点击 **Create** -> **New Notebook**。
2. 在右侧设置面板（Notebook options）：
   - **Accelerator**：选择 **GPU T4 x2**（或 **GPU T4**）。
   - **Internet**：开启（**Internet on**，下载依赖和预训练权重必须）。

### 步骤 2：挂载 VisDrone 数据集
1. 在右侧面板点击 **+ Add Input**。
2. 在搜索框输入：`visdrone`。
3. 选择收录有 `VisDrone2019-DET` 的公开数据集（如 `bhavikjikadara/visdrone-dataset` 或 `msherievan/visdrone2019`），点击添加。
   > 挂载后，数据位于 `/kaggle/input/` 中，Notebook 脚本会自动检索识别。

### 步骤 3：导入代码与 Notebook
**方式 1（最简单推荐）**：
- 点击 Kaggle Notebook 顶部菜单 **File** -> **Import Notebook**。
- 选择本地的 `D:\cv\MambaCenter-cpia\notebooks\MambaCenter_Kaggle_Training.ipynb` 上传打开。

**方式 2（通过 GitHub 私有/公有仓库拉取）**：
- 如果你的项目已在 GitHub `hurry211/MambaCenter-Research`：
  - 若为**公开仓库**，Notebook 会直接 `git clone`。
  - 若为**私有仓库**，请在 Kaggle 顶部菜单 **Add-ons** -> **Secrets** 中添加名为 `GITHUB_PAT` 的密钥（填入你的 GitHub Personal Access Token），Notebook 会自动进行带鉴权的无感拉取。

### 步骤 4：一键执行训练流水线

打开 Notebook 后，从上到下逐单元格运行（或直接点击 **Run All**）：

1. **环境检查**：确认识别到 `Tesla T4`，自动检测 FP16 加速。
2. **克隆代码与装包**：安装 `ninja`、`pycocotools`、`timm` 等。
3. **编译 Selective Scan 扩展**：
   - 执行 `python runtime/selective-scan/build.py build_ext ...`。
   - 耗时约 2 分钟，提示 `[OK] selective_scan_cuda extension compiled and imported successfully!` 即编译成功。
4. **下载 DLA-34 预训练权重**：自动拉取并缓存至 `/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth`。
5. **数据预处理**：自动建立软链接，调用 `visdrone2coco.py` 转换 train 与 val 的 COCO 标注格式（耗时约 30 秒）。
6. **单元测试与冒烟测试**：
   - 先跑 25 项单元测试与 GPU 算子验证。
   - 运行 10 步冒烟测试确认显存占用稳定在 8–10 GB，无 OOM 风险。
7. **正式训练**：
   ```bash
   python tools/train.py --config configs/kaggle_addon_cp.yaml --out /kaggle/working/runs/cpmamba_kaggle_t4_s317
   ```
8. **评测与画图**：
   - 自动在 validation 集评测 COCO AP 指标。
   - 自动读取 `history.jsonl` 绘制损失曲线。
   - 自动在验证集图片上画出检测目标框并展示。

---

## 四、训练过程技巧与断点续训

- **中断后恢复训练**：
  若 Kaggle 会话中断或超时，重新启动会话后，直接添加 `--resume` 即可从上一次保存的 `model_last.pth` 平滑续训：
  ```bash
  python tools/train.py --resume --out /kaggle/working/runs/cpmamba_kaggle_t4_s317
  ```
- **切换模型架构**：
  - CP-Mamba（B5'）：`--config configs/kaggle_addon_cp.yaml`
  - IA-Mamba（A4'）：`--config configs/kaggle_addon_ia.yaml`
  - 原版 CenterNet 基线：`--config configs/kaggle_t4.yaml`
- **下载产物**：
  训练结束后，Kaggle 右侧面板的 `/kaggle/working/runs` 目录下可以直接下载 `model_final.pth`、`eval_val.json` 和 `history.jsonl`。

