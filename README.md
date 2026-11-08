# MambaCenter · VisDrone 小目标检测

本项目在 CenterNet 上研究 Mamba 模块，维护以下模型：
- 原版 CenterNet（DLA-34 + DCNv2）基线；
- 两个 idea（CP-Mamba、IA-Mamba）的两种接法：
  - 第一批：用 Mamba-FPN 替换 DCN 颈部；
  - 第二批：保留 DCN 颈部，作为零初始化外挂；
- 独立的官方 Mamba-YOLO-B 对照。

## 目录

```text
MambaCenter/
├── baseline/CenterNet/     原版 CenterNet（DLA-34 + DCNv2）及 VisDrone / PyTorch 2 / K=500 补丁；
│                           所有 CenterNet 系模型共用它的数据增强、CtdetLoss、解码与后处理
├── mambacenter/            两个 idea 的模型代码
│   ├── scan.py             selective-scan 后端与扫描路由（光栅、优先序、稀疏选择）
│   ├── vss.py              普通 VSS 块（四向 SS2D）
│   ├── fpn.py              Mamba-FPN 自顶向下颈部（P5→P2）
│   ├── cp_mamba.py         CP-Mamba：粗热图、中心先验、CP-SSM
│   ├── ia_mamba.py         IA-Mamba：创新量、局部归一化、四向软最小、IA-VSS
│   ├── detector.py         DLA-34 + Mamba 颈部 + CenterNet 头（第一批）
│   ├── addon.py            原版 DCN 颈部 + stride 8/4 零初始化残差外挂（第二批）
│   └── centernet.py        注册为 CenterNet 架构 mambafpn_34 / cpmamba_34 / iamamba_34 /
│                           dlavss_34 / dlacp_34 / dlaia_34
├── configs/                实验配置：_base.yaml 为公共训练方案，其余每个实验一份
├── tools/                  train / evaluate / probe（显存探测）/ visdrone2coco / visualize
├── tests/                  CPU 单元测试与 GPU 训练前检查
├── slurm/                  集群作业脚本（本地保留，不发布）
├── mamba-yolo/             官方 Mamba-YOLO-B（保留，含原结果）
├── runtime/selective-scan/ mamba_ssm selective_scan_cuda 源码与构建
├── docs/                   环境配置、实验方案、两个 idea 的实现说明
└── results/                comparison.json 对照索引
```

数据、权重、二进制扩展和实验输出不在仓库中；集群上实验目录为 `$MC_RUNS/<name>_s<seed>`。

## 模型

| 配置 | arch | 说明 |
|---|---|---|
| [centernet_dla34](configs/centernet_dla34.yaml) | dla_34 | 原版 CenterNet：DLAUp/IDAUp 中的 DCNv2（与原 DCNv2 数值一致，见 tests/test_dcn.py），19.66M |
| [mambafpn](configs/mambafpn.yaml) | mambafpn_34 | DLA-34 + 普通 VSS Mamba-FPN，两个 idea 的直接基线（A1），16.89M |
| [cpmamba_b1](configs/cpmamba_b1.yaml) | cpmamba_34 | Mamba-FPN + P3 粗热图辅助监督（B1），17.04M |
| [cpmamba](configs/cpmamba.yaml) | cpmamba_34 | [CP-Mamba](docs/CP-Mamba.md)（B5），17.03M |
| [iamamba](configs/iamamba.yaml) | iamamba_34 | [IA-Mamba](docs/IA-Mamba.md)（A4），17.29M |
| [addon_vss](configs/addon_vss.yaml) | dlavss_34 | 第二批对照（A1′）：原版 DCN 颈部 + stride 8/4 各 2 个零初始化 VSS 块（含 FFN），20.38M（新增 0.73M） |
| [addon_cp_b1](configs/addon_cp_b1.yaml) | dlacp_34 | A1′ + stride-8 粗热图辅助监督（B1′），20.53M |
| [addon_cp](configs/addon_cp.yaml) | dlacp_34 | CP-Mamba 外挂（B5′）：stride 4 为 CP-SSM，20.52M |
| [addon_ia](configs/addon_ia.yaml) | dlaia_34 | IA-Mamba 外挂（A4′）：两处都是 IA-VSS，创新图入热图头，20.53M |
| [Mamba-YOLO-B](mamba-yolo/README.md) | — | 官方结构，独立代码与训练方案 |

## 开始使用

1. 按 [环境配置](docs/环境配置.md) 准备 PyTorch、selective-scan 扩展和 VisDrone 数据（`data/visdrone` → VisDrone2019-DET）。
2. 运行检查：`python -m unittest discover -s tests -p 'test_*.py'`（CPU），`python tests/gpu_checks.py --out <json>`（GPU）。
3. 训练与评测：

   ```bash
   python tools/train.py --config configs/cpmamba.yaml
   python tools/evaluate.py --run $MC_RUNS/cpmamba_s317
   ```

每个新实验写入新的运行目录；续训显式使用 `--resume --out <run>`。训练方案与参数依据见 [实验方案](docs/实验方案.md)。

## 当前结果与实验基准

### 1. 结果来源与协议说明

- **上游公布基准（140 轮，seed 317）**：来自上游仓库 `lexiangrui/MambaCenter`（基准提交 `6c44b90`），系上游集群单种子公布结果，未经本地重新评测前仅作为参考基线，不记为独立复现。
- **工程适配运行（15 轮，Tesla T4）**：来自 `hurry211/MambaCenter` 在 Kaggle Tesla T4 (16GB) 上的工程适配与梯度累加短训（`dlacp_34`，保留 DCN 的 CP 外挂方案）。该运行证明了工程栈可在消费级/免费 GPU 上稳定训练，但指标随轮次上涨属于正常优化收敛，**不能**直接解释为 CP 机制单独带来的增益。确定 CP 机制贡献需依赖同等条件的 R0/R1/R2 对照实验。
- **评测协议与口径区分**：
  - 本文采用转换后的 VisDrone2019-DET val（548 张）COCO 格式标注，使用 pycocotools 的标准 `COCOeval(..., 'bbox')` 计算指标。
  - 标准指标报告包含：$\text{AP}$ (mAP@[.50:.95])、$\text{AP}_{50}$、$\text{AP}_{75}$、小目标 $\text{AP}_s$、中目标 $\text{AP}_m$、大目标 $\text{AP}_l$ 及召回率 $\text{AR}_{100}$。
  - 检出数量上限：标准 COCO 使用 $\text{maxDets}=100$，针对 VisDrone 高密场景同时评测并记录 $\text{maxDets}=500$。
  - **重要说明**：此协议属于“COCO-style 评测”，**不等同于** VisDrone 官方包含忽略区域（ignored regions / ignore areas）过滤的 MATLAB/C++ 评测基准。

### 2. 上游仓库公布结果（140 轮全周期参考，seed 317）

> [!NOTE]
> 以下为上游仓库公布的最后一轮（Epoch 140）指标，仅用于宏观参考，尚未完成多种子独立复现。

| 模型 | arch | 轮数 | AP@100 (%) | AP50@100 (%) | 状态说明 |
|---|---|---:|---:|---:|---|
| CenterNet DLA-34 (DCNv2) | `dla_34` | 140 | 26.40 | 47.45 | 上游公布 baseline，seed 317 |
| Mamba-FPN（A1，替换 DCN 颈部） | `mambafpn_34` | 140 | 23.65 | 44.71 | 上游公布，neck 替换方案 |
| CP-Mamba B1（Mamba-FPN + 粗热图） | `cpmamba_34` | 140 | 23.35 | 44.20 | 上游公布，neck 替换方案 |
| CP-Mamba B5（完整 CP 颈部） | `cpmamba_34` | 140 | 23.26 | 44.13 | 上游公布，neck 替换方案 |
| IA-Mamba（A4，完整 IA 颈部） | `iamamba_34` | 140 | 23.80 | 44.91 | 上游公布，neck 替换方案 |
| 外挂 VSS（A1′，保留 DCN 外挂） | `dlavss_34` | 140 | 27.34 | 48.17 | 上游公布第二批，seed 317 |
| 外挂 CP B1′（A1′ + 粗热图监督） | `dlacp_34` | 140 | 27.68 | 48.79 | 上游公布第二批，seed 317 |
| 外挂 CP B5′（完整 CP 外挂） | `dlacp_34` | 140 | 27.95 | 48.97 | 上游公布第二批，seed 317 |
| 外挂 IA（A4′，有下界） | `dlaia_34` | 140 | 27.77 | 48.81 | 上游公布第二批，seed 317 |
| 外挂 IA A5′（去掉下界） | `dlaia_34` | 140 | 27.98 | 49.39 | 上游公布第二批，seed 317 |
| Mamba-YOLO-B | — | 75 | 30.51 | 49.65 | 官方结构，300 轮计划中途取消保留值 |

### 3. CP-Mamba 外挂方案的 T4 训练与工程适配实测（15 轮逐轮验证）

在 Kaggle Tesla T4 (16GB) 上使用 `configs/kaggle_addon_cp.yaml`（`dlacp_34`，batch 8 × accum 2 = 等效 batch 16，lr 6.25e-5，amp fp16）完成 15 轮逐轮真实评测（`eval.interval=1`）。

以下数据全部提取自原始运行记录 `log.txt` 与 `history.jsonl`（真实无截断）：

| 轮次 (Epoch) | 训练 Loss | AP@100 | AP@500 | AP50@100 | AP50@500 | AP75@100 | AP75@500 | APs | APm | APl | AR100 | 单轮耗时 | 显存峰值 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 7.8363 | 3.03% | 3.03% | 9.66% | 9.66% | 1.10% | 1.10% | 1.6% | 5.2% | 5.1% | 11.5% | 77.5m | 8.15 GiB |
| **Epoch 2** | 2.4730 | 7.36% | 7.36% | 18.37% | 18.39% | 5.11% | 5.11% | 3.7% | 12.0% | 14.2% | 19.3% | 77.4m | 8.15 GiB |
| **Epoch 3** | 2.1867 | 9.17% | 9.18% | 23.08% | 23.10% | 6.16% | 6.16% | 4.6% | 15.2% | 19.3% | 22.2% | 77.3m | 8.15 GiB |
| **Epoch 4** | 2.0213 | 11.17% | 11.17% | 25.39% | 25.43% | 8.73% | 8.73% | 6.0% | 17.9% | 23.2% | 25.9% | 77.3m | 8.15 GiB |
| **Epoch 5** | 1.9466 | 12.61% | 12.62% | 28.07% | 28.10% | 9.87% | 9.88% | 6.6% | 19.7% | 26.8% | 26.7% | 77.4m | 8.15 GiB |
| **Epoch 6** | 1.8618 | 12.64% | 12.64% | 27.73% | 27.77% | 10.51% | 10.51% | 6.2% | 20.5% | 26.9% | 26.0% | 78.4m | 8.27 GiB |
| **Epoch 7** | 1.8005 | 14.22% | 14.23% | 31.70% | 31.74% | 11.16% | 11.16% | 8.3% | 21.4% | 26.6% | 29.4% | 78.4m | 8.27 GiB |
| **Epoch 8** | 1.7617 | 15.04% | 15.06% | 32.41% | 32.47% | 12.37% | 12.40% | 9.1% | 22.8% | 28.1% | 30.3% | 78.5m | 8.27 GiB |
| **Epoch 9** | 1.7141 | 16.30% | 16.31% | 34.84% | 34.89% | 13.34% | 13.34% | 9.6% | 24.6% | 28.7% | 31.7% | 72.1m | 8.27 GiB |
| **Epoch 10** | 1.6696 | 16.33% | 16.34% | 34.62% | 34.66% | 13.57% | 13.57% | 9.8% | 25.3% | 32.2% | 32.5% | 71.8m | 8.27 GiB |
| **Epoch 11** | 1.6316 | 16.62% | 16.64% | 35.92% | 35.96% | 13.54% | 13.55% | 10.1% | 24.7% | 30.2% | 32.7% | 71.9m | 8.27 GiB |
| **Epoch 12** | 1.5849 | 17.30% | 17.32% | 36.03% | 36.10% | 14.66% | 14.66% | 10.0% | 26.7% | 34.5% | 32.8% | 71.9m | 8.27 GiB |
| **Epoch 13** | 1.5633 | 18.02% | 18.03% | 37.49% | 37.54% | 14.86% | 14.87% | 10.6% | 27.0% | 34.1% | 34.2% | 71.9m | 8.27 GiB |
| **Epoch 14** | 1.5327 | 18.25% | 18.26% | 37.70% | 37.76% | 15.79% | 15.79% | 10.9% | 26.8% | 35.1% | 34.3% | 72.0m | 8.27 GiB |
| **Epoch 15 (最终)** | **1.5100** | **19.04%** | **19.06%** | **39.46%** | **39.51%** | **16.00%** | **16.00%** | **11.9%** | **27.2%** | **37.4%** | **35.8%** | **72.0m** | **8.27 GiB** |

- **最佳指标（Best Checkpoint）**：Epoch 15 即为当前 15 轮的最优轮次（Final == Best）。
- **产物索引**：
  - 收敛曲线图：`results/loss_curve_15epochs.png`
  - 逐轮指标记录：`results/history_15epochs.jsonl`
  - 最终评测输出：`results/eval_val_15epochs.json`
  - 验证集样张检测结果：`results/vis/`

### 4. 受控基线与消融实验矩阵（待云端授权运行）

为消除仅由轮次增加引起的假象，并严格量化各机制贡献，已构建完全对齐的 T4 实验配置：

| 对照编号 | 模型与配置 | arch | 总参数量 | 核心机制与参数设定 | 验证目的 |
|---|---|---|---:|---|---|
| **R0** | [kaggle_centernet_dla34](configs/kaggle_centernet_dla34.yaml) | `dla_34` | 19.658M | 原版 DLA-34 + DCNv2 | 基准底线 |
| **R1** | [kaggle_addon_vss](configs/kaggle_addon_vss.yaml) | `dlavss_34` | 20.383M | DCN + 普通 VSS 外挂（A1′） | $R1 - R0$：评估普通 Mamba 外挂整体增益 |
| **R2** | [kaggle_addon_cp_b1](configs/kaggle_addon_cp_b1.yaml) | `dlacp_34` | 20.532M | VSS 外挂 + stride-8 粗热图辅助监督 (`use_cp_block: false`) | $R2 - R1$：评估粗热图辅助监督的独立增益 |
| **R3** | [kaggle_addon_cp](configs/kaggle_addon_cp.yaml) | `dlacp_34` | 20.524M | 完整 CP 外挂（Δ 调制 + 优先排序 + 稀疏扫描，B5′） | $R3 - R2$：评估完整 CP 机制组合的额外价值 |
| **R4** | [kaggle_addon_ia_a5](configs/kaggle_addon_ia_a5.yaml) | `dlaia_34` | 20.527M | IA-Mamba 外挂（无下界，A5′） | IA 对照方案 |
| **CP-Abl1** | [kaggle_addon_cp_nodelta](configs/kaggle_addon_cp_nodelta.yaml) | `dlacp_34` | 20.524M | 关闭 Δ 调制 (`delta_mod: false`) | 验证先验抬升 Δ 的实际价值 |
| **CP-Abl2** | [kaggle_addon_cp_raster](configs/kaggle_addon_cp_raster.yaml) | `dlacp_34` | 20.524M | 保留选点，纯光栅顺序 (`bins: 1`)，无 Δ 调制 | 隔离选点与扫描优先序的贡献 |
| **IA-Abl** | [kaggle_addon_ia_ablation](configs/kaggle_addon_ia_ablation.yaml) | `dlaia_34` | 20.522M | 关闭 Δ 调制、不入热图头、辅助损失权为 0 | 隔离 IA 创新图与辅助损失的影响 |

机器可读索引：[results/comparison.json](results/comparison.json)。

## 上游来源

CenterNet、DCNv2、Mamba、Mamba-YOLO 的源码和许可证保留在各自目录，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
