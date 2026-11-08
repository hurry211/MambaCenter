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

## 当前结果

VisDrone2019-DET val（548 张），转换后的 COCO 标注，整体 AP/AP50 使用 maxDets=100，不等同于 VisDrone 官方忽略区域评测。CenterNet 系结果同时记录 maxDets=500。

### 1. 140 轮全周期基准（集群基线）

| 模型 | 轮数 | AP (%) | AP50 (%) | 状态 |
|---|---:|---:|---:|---|
| CenterNet DLA-34 (DCNv2) | 140 | 26.40 | 47.45 | seed 317，最后一轮 |
| Mamba-FPN（A1） | 140 | 23.65 | 44.71 | seed 317，最后一轮 |
| CP-Mamba B1（A1 + 粗热图监督） | 140 | 23.35 | 44.20 | seed 317，最后一轮 |
| CP-Mamba B5（完整） | 140 | 23.26 | 44.13 | seed 317，最后一轮 |
| IA-Mamba（A4） | 140 | 23.80 | 44.91 | seed 317，最后一轮 |
| 外挂 VSS（A1′） | 140 | 27.34 | 48.17 | 第二批，seed 317，最后一轮 |
| 外挂 CP B1′（A1′ + 粗热图监督） | 140 | 27.68 | 48.79 | 第二批，seed 317，最后一轮 |
| 外挂 CP B5′（完整） | 140 | 27.95 | 48.97 | 第二批，seed 317，最后一轮 |
| 外挂 IA（A4′） | 140 | 27.77 | 48.81 | 第二批，seed 317，最后一轮 |
| 外挂 IA A5′（去掉下界） | 140 | 27.98 | 49.39 | 第二批，seed 317，最后一轮 |
| Mamba-YOLO-B | 75 | 30.51 | 49.65 | 300 轮计划在 75 轮被取消，保留原结果 |

### 2. 15 轮云端快速验证协议（Fast-Validation on Tesla T4）

在 Kaggle Tesla T4 (16GB) 上使用 `configs/kaggle_addon_cp.yaml`（CP-Mamba 外挂，dlacp_34）执行 15 轮逐轮官方评测（`eval.interval=1`）：

| 阶段 / 轮次 | 训练平均 Loss | $\text{AP}_{50}^{500}$ | $\text{AP}$ (mAP@[.50:.95]) | $\text{AP}_{75}$ | 小目标 $\text{AP}_s$ | 中目标 $\text{AP}_m$ | 大目标 $\text{AP}_l$ | $\text{AR}_{100}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 7.836 | 9.66% | 3.03% | 2.15% | 1.2% | 5.8% | 7.2% | 12.1% |
| **Epoch 5** | 1.947 | 28.91% | 12.70% | 10.15% | 7.6% | 21.1% | 27.2% | 27.9% |
| **Epoch 8** | 1.762 | 32.47% | 15.06% | 12.40% | 9.1% | 24.6% | 31.5% | 30.3% |
| **Epoch 11** | 1.632 | 35.96% | 16.64% | 13.55% | 10.1% | 24.7% | 30.2% | 32.7% |
| **Epoch 12** | 1.585 | 36.10% | 17.32% | 14.66% | 10.0% | 26.7% | 34.5% | 32.8% |
| **Epoch 13** | 1.563 | 37.54% | 18.03% | 14.87% | 10.4% | 26.5% | 34.8% | 33.9% |
| **Epoch 14** | 1.533 | 37.76% | 18.26% | 15.79% | 10.9% | 26.8% | 35.1% | 34.3% |
| **Epoch 15 (最终)** | **1.510** | **39.51%** | **19.06%** | **16.00%** | **11.9%** | **27.2%** | **37.4%** | **35.8%** |

* 15 轮高精收敛曲线图：`results/loss_curve_15epochs.png`
* 逐轮详细指标：`results/history_15epochs.jsonl`
* 最终评测输出：`results/eval_val_15epochs.json`
* 验证集目标检测实测样张：`results/vis/`

CenterNet 系每个模型只有一个 seed，seed 间波动尚未估计。maxDets=500 的数值见 comparison.json，与 maxDets=100 相差不到 0.1。

机器可读索引：[results/comparison.json](results/comparison.json)。

## 上游来源

CenterNet、DCNv2、Mamba、Mamba-YOLO 的源码和许可证保留在各自目录，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
