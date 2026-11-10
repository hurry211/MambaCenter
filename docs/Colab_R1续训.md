**Colab R1 续训入口｜2026-11-10 修订**

Notebook 的默认任务是 `R1_RESUME`，继续已验证的 Kaggle R1 第8轮检查点，目标第15轮。R2 仍是之后独立运行的粗热图对照，不会在 R1 后自动启动。

1. 在 Colab 打开修订后的 `notebooks/MambaCenter_Colab_Training.ipynb`，选择 T4 GPU。
2. 挂载自己的 Google Drive。输入放在 `MyDrive/MambaCenter_Colab_Inputs/`：完整 `r1_epoch8_checkpoint.zip`，或四个分块及 `r1_epoch8_checkpoint.manifest.json`。预训练文件 `dla34-ba72cf86.pth` 可放在同一目录；否则尝试下载并核对历史文件哈希。
3. 依次运行准备单元。Notebook 会固定检出训练源码 `d6c1f07dc35e85e5f077b90f7f9b8c7e9a8d62d6`，验证断点、数据标注指纹及原生扫描/DCN。准备失败会停止，不打印“全部就绪”掩盖失败。
4. 第10阶段才开始训练。正式输出直接写入 `MyDrive/MambaCenter_Checkpoints/vss_colab_t4_s317/`，每轮检查点和日志由训练器直接落盘；并非等整个进程退出后再复制。训练图像放在 Colab 临时盘。
5. 新会话重新挂载、准备代码/数据，再使用同一任务和输出目录。已有本任务第9–14轮完整检查点时从它继续；已完成第15轮则不重复启动。如果检查点保存后、该轮评价完成前断开，检查点与history会不一致，Notebook会保留文件并停止，需补回评价后继续，不能静默跳轮或重跑覆盖。

首次 R1 检查点的 SHA256 必须为 `7ac0169203b51ca35419b651d340d4b7ec2ffb583c150f66f822f939c7744b12`，epoch=8、step=3232，包含模型、优化器和 FP16 scaler。不会在整个输入目录里挑第一个 `.pth`，也不只检查epoch后把打印哈希当校验。

结果限制作如实记录：Kaggle历史检查点没有完整RNG/worker状态；切换平台、Torch/CUDA版本及运行时会话后不能称为逐步完全复现。保存环境、源码版本、数据指纹和恢复边界，结果按跨会话探索性实验报告。原生GPU数值检查仍须在实际Colab运行时执行；CPU检查不替代它。

免费Colab不保证特定GPU、资源额度和运行时长，不能承诺单会话完成约9–10小时的R1剩余训练或约20小时的R2。直接写云盘可保留已成功写入的产物，仍不能承诺数据永不丢失或恢复最后一个未完成batch。参见 [Colab官方FAQ](https://research.google.com/colaboratory/faq.html)。

生成入口为 `tools/build_colab_notebook.py`，会把 `tools/colab_support.py` 内嵌进Notebook；本地只运行生成、语法和CPU存储/校验测试，不运行正式训练。
