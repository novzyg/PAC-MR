# PAC-MR

患者条件冲突感知药物推荐的研究代码。训练分为两个阶段：首先学习基础表示与预测头，再冻结基础网络、学习冲突调整模块。

## 项目结构

```text
PAC-MR/
├── src/
│   ├── main.py             # 命令行入口
│   ├── trainer.py          # 训练、验证、测试与数据概览
│   ├── models.py           # 基础模型与冲突调整模块
│   ├── data.py             # 数据加载、划分与批处理
│   └── metrics.py          # 指标与验证选择规则
├── configs/
│   ├── grid.json           # 多 GPU 网格搜索配置
│   └── sensitivity.json    # 单因素实验配置
├── scripts/
│   ├── train.sh            # 连续执行两阶段训练
│   ├── evaluate.sh         # 独立测试入口
│   ├── env.sh              # Shell 脚本公共环境
│   ├── grid_search.py      # 多 GPU 训练调度
│   ├── sensitivity.py      # 单因素实验与汇总
│   └── tune_threshold.py   # 基于验证结果选择阈值
├── tests/
│   └── test_model.py       # 合成数据模型检查
├── requirements.txt
└── .gitignore
```

下文命令均从仓库根目录执行。数据与运行产物不随仓库发布；`data/` 由用户准备，`saved/` 在运行时生成。

## 安装

历史验证使用 Python 3.10、PyTorch 2.5.1。当前依赖文件保留原项目版本范围，并非完全锁定的环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 训练需使用与本机环境兼容的 PyTorch。Shell 脚本使用当前环境的 `python3`，可通过 `PYTHON_BIN` 指定解释器。

## 数据准备

自行准备 SafeDrug 风格的预处理数据，每个数据集目录需包含：

| 文件 | 内容 |
| --- | --- |
| `records_final.pkl` | 按患者组织的就诊记录，每次就诊包含诊断、操作、药物索引 |
| `voc_final.pkl` | 诊断、操作、药物词表 |
| `ddi_A_final.pkl` | 药物相互作用邻接矩阵 |

默认位置为 `data/mimic-iv_all_all/`，也可指定外部数据目录。本仓库不包含原始数据或数据预处理流水线；不同预处理版本不能直接视为同一实验设置。

默认按原始患者顺序划分：前 2/3 为训练集，其后剩余的一半为测试集，最后为验证集。共现图仅使用训练患者构建。药物标签不作为推理输入。

## 训练与测试

在同一终端中设置实际路径：

```bash
export DATA_DIR=/absolute/path/to/dataset
export OUTPUT_DIR="$PWD/saved/run_001"
export DEVICE=cuda:0
bash scripts/train.sh
bash scripts/evaluate.sh
```

`train.sh` 先训练 `base`，再以其最佳检查点初始化 `full`。测试独立执行，读取 `full/best.pt` 保存的验证阈值，不使用测试集选择模型。

单配置脚本可通过 `DIM`、`GRAPH_LAYERS`、`ATTENTION_HEADS`、`BASE_EPOCHS`、`ADJUST_EPOCHS`、`SEED`、`DDI_WEIGHT` 设置参数。额外参数传给两个训练阶段，例如：

```bash
bash scripts/train.sh --dropout 0.1 --batch-size 4 --eval-batch-size 8
```

`dim` 必须能被 `attention_heads` 整除。默认单配置训练的基础/调整学习率为 0.0003/0.001，最大轮数为 30/20；网格和单因素实验使用各自 JSON 配置，默认参数并不完全相同。

完整命令行选项：

```bash
python src/main.py --help
python src/main.py train --help
python src/main.py evaluate --help
```

`prepare` 子命令仅生成数据划分与概览，不负责原始数据预处理：

```bash
python src/main.py prepare --data "$DATA_DIR" --output saved/data_summary
```

## 网格搜索

编辑 `configs/grid.json` 的 `fixed` 和 `grid` 字段。当前配置包含 12 组组合，各运行两个训练阶段；每张卡同时运行一个任务。

```bash
python scripts/grid_search.py --dry-run
python scripts/grid_search.py --data "$DATA_DIR" --gpus 0,1 --output saved/grid_v1
```

查看 `results.csv` 中的验证结果，选定配置后单独测试：

```bash
python src/main.py evaluate --run saved/grid_v1/run_实际编号/full --device cuda:0
```

Python 工具通过命令行接收参数；设置 `DATA_DIR` 或 `DEVICE` 环境变量不会自动覆盖它们的默认值。

## 单因素实验

`configs/sensitivity.json` 定义 DDI 权重、位移上限、成本比例的单因素扫描及评估阈值。同种子共享一个基础模型；默认 3 个种子、42 次调整训练和 6 个阈值。

```bash
python scripts/sensitivity.py train --dry-run
python scripts/sensitivity.py train --data "$DATA_DIR" --device cuda:0 --output saved/sensitivity_v1
python scripts/sensitivity.py test --data "$DATA_DIR" --device cuda:0 --output saved/sensitivity_v1
```

训练和测试需保持相同的配置、种子、数据与输出目录。可用 `--config` 指定配置，`--seeds 1023` 运行单个种子。

各模型按固定阈值 0.5 的验证 Jaccard 选择检查点，然后复用预测评估其他阈值；并非为每个阈值分别选择最佳轮次。顶层 `val_thresholds.csv`、`test_thresholds.csv` 保存详细指标，`val_summary.csv`、`test_summary.csv` 保存跨种子汇总。

## 验证阈值选择

以下工具从已保存的验证曲线中，在 Jaccard 满足要求的候选中选择 DDI 最低者，不更新权重、不使用测试集选参数。该设置应与固定阈值实验区分报告。

```bash
python scripts/tune_threshold.py --source saved/grid_v1 \
  --output saved/threshold_tuned/full --min-jaccard 0.33 --margin 0.005
python src/main.py evaluate --run saved/threshold_tuned/full --device cuda:0
```

## 检查与运行产物

无需正式数据的模型自检（需安装依赖）：

```bash
python tests/test_model.py
```

检查覆盖降分方向、冻结梯度、标签输入隔离、padding、空编码、训练集构图、参数边界及权重重载。小规模两阶段流程检查：

```bash
DATA_DIR="$DATA_DIR" DEVICE=cpu OUTPUT_DIR="$PWD/saved/smoke" \
  BASE_EPOCHS=1 ADJUST_EPOCHS=1 bash scripts/train.sh --smoke
DEVICE=cpu OUTPUT_DIR="$PWD/saved/smoke" bash scripts/evaluate.sh
```

`--smoke` 仅使用前 6 名患者并限制训练步数，不代表论文性能。正式训练需使用新输出目录。

运行目录保存 `config.json`、`environment.json`、`best.pt`、`last.pt`、验证预测和诊断；测试另存 `test_metrics.json`、`test_predictions.npz` 等。单次测试拒绝覆盖已有结果。训练脚本支持断点恢复，但配置、源码或数据改变后必须使用新输出目录，不应删除锁文件或修改指纹绕过检查。

## 复现说明

- Jaccard、F1、Precision、Recall 和 AP 按患者平均；DDI 为全局药物对比例。字段 `prauc` 实际使用 `average_precision_score`。
- 调整阶段始终冻结基础网络，`--freeze-epochs` 为遗留参数；`--layers` 仅支持 1。
- 当前历史注意力包含当前就诊；事件注意力为线性打分，成本网络隐藏维度为 32。描述模型时应以实现为准。
- 当前代码已修复位移上限和成本比例未传入训练模型的问题。缺少参数生效标记的旧非默认配置检查点会被拒绝；旧相关敏感性结果需重新验证。
- 整理改变了源码文件名与指纹，不支持直接续训整理前的输出目录。旧检查点还可能依赖原服务器的绝对数据路径。
- 仓库目前未附可核验的正式论文结果或预训练权重。论文元信息、引用及许可证待作者补充。
