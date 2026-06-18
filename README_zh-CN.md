# Pipette

### 面向湿实验室机器人的具身仿真平台、评测基准与数据高效增强框架

[English](README.md) | [简体中文](README_zh-CN.md)

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-76B900)](https://isaac-sim.github.io/IsaacLab/)
[![LeRobot](https://img.shields.io/badge/LeRobot-ACT%20%7C%20SmolVLA%20%7C%20PI0-yellow)](https://github.com/huggingface/lerobot)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)

**Pipette** 是一个面向湿实验室机器人学习的具身仿真平台。项目将可编辑实验室资产、语言引导任务注册、遥操作数据采集、成功验证仿真增强、LeRobot 数据转换、VLA 策略训练和闭环评测整合为统一工作流。

## 平台总览

<p align="center">
  <img src="./docs/figures/pipette_overview.png" alt="Pipette 平台总览" width="100%">
</p>

## 主要特点

- **可编辑湿实验室资产：** 采用 USD/USDZ 格式统一表示几何、材质、碰撞体、物理属性和任务语义。
- **统一的 11 任务基准：** 覆盖样品处理、培养耗材操作、设备开合、精确放置和物体转移。
- **物理一致的数据增强：** 在 Isaac Sim 中重新执行轨迹，并施加光照、相机、速度和动作扰动。
- **自动成功验证：** 使用任务级评估器筛选增强 episode，并记录可解释的失败原因。
- **适配 VLA 的数据链路：** 将同步的多视角图像、机器人状态、动作和语言指令从 HDF5 转换为 LeRobot 数据格式。
- **统一策略评测：** ACT、SmolVLA 和 PI0 通过相同的 ZMQ 接口连接 Isaac Lab 评测环境。
- **自然语言 Agent：** 通过命令行或网页完成环境搭建、任务注册、数据采集、增强、训练和评测。

## 基准任务

### 示例视频

<table>
  <tr>
    <td align="center"><img src="./docs/gif/Pick%20up%20the%20test%20tube.gif" alt="拿起试管" width="100%"><br><sub>拿起试管</sub></td>
    <td align="center"><img src="./docs/gif/Position%20the%20pipette%20over%20the%20petri%20dish.gif" alt="将移液枪定位到培养皿上方" width="100%"><br><sub>将移液枪定位到培养皿上方</sub></td>
    <td align="center"><img src="./docs/gif/Remove%20the%20petri%20dish%20from%20the%20incubator.gif" alt="从培养箱中取出培养皿" width="100%"><br><sub>从培养箱中取出培养皿</sub></td>
    <td align="center"><img src="./docs/gif/Place%20the%20petri%20dish%20in%20the%20incubator.gif" alt="将培养皿放入培养箱" width="100%"><br><sub>将培养皿放入培养箱</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="./docs/gif/Close%20the%20centrifuge%20lid.gif" alt="关闭离心机盖" width="100%"><br><sub>关闭离心机盖</sub></td>
    <td align="center"><img src="./docs/gif/Open%20the%20centrifuge%20lid.gif" alt="打开离心机盖" width="100%"><br><sub>打开离心机盖</sub></td>
    <td align="center"><img src="./docs/gif/Open%20the%20water%20bath%20lid.gif" alt="打开水浴锅盖" width="100%"><br><sub>打开水浴锅盖</sub></td>
    <td align="center"><img src="./docs/gif/Close%20the%20spectrophotometer%20lid.gif" alt="关闭分光光度计盖" width="100%"><br><sub>关闭分光光度计盖</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="./docs/gif/Place%20the%20centrifuge%20tube%20on%20the%20balance.gif" alt="将离心管放到电子天平上" width="100%"><br><sub>将离心管放到电子天平上</sub></td>
    <td align="center"><img src="./docs/gif/Remove%20the%20centrifuge%20tube%20from%20the%20balance.gif" alt="从电子天平取下离心管" width="100%"><br><sub>从电子天平取下离心管</sub></td>
    <td align="center"><img src="./docs/gif/Place%20the%20pipette%20on%20the%20pipette%20stand.gif" alt="将移液枪放到移液枪架上" width="100%"><br><sub>将移液枪放到移液枪架上</sub></td>
    <td></td>
  </tr>
</table>

| 样品与培养耗材 | 设备开合操作 | 仪器放置与转移 |
|---|---|---|
| 拿起试管 | 关闭离心机盖 | 将离心管放到电子天平上 |
| 将移液枪定位到培养皿上方 | 打开离心机盖 | 从电子天平取下离心管 |
| 从培养箱中取出培养皿 | 打开水浴锅盖 | 将移液枪放到移液枪架上 |
| 将培养皿放入培养箱 | 关闭分光光度计盖 | |

## 环境安装

### 1. 安装项目与依赖

Pipette 使用两个相互独立的 Conda 环境：

- `env_isaaclab`：安装 NVIDIA Isaac Sim 5.1.0 和 Isaac Lab 2.3.2
- `lerobot`：安装支持 ACT、SmolVLA 和 PI0 的 LeRobot

将仿真环境和策略训练环境分开，可以避免 Python 版本和二进制依赖冲突。

以下命令面向 Linux x86_64。请先安装 Isaac Sim 和 Isaac Lab，再安装 LeRobot，最后克隆 Pipette。

#### 安装 Isaac Sim 5.1 和 Isaac Lab 2.3.2

Isaac Lab 2.3.2 提供官方 pip 软件包，可以同时安装 Isaac Lab 组件和 Isaac Sim 5.1：

```bash
#安装isaacsim和isaaclab环境
conda create -y -n env_isaaclab python=3.11
conda activate env_isaaclab
python -m pip install --upgrade pip

python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com
python -m pip install h5py pyzmq
```


#### 安装 LeRobot

使用 Python 3.12 创建独立环境，克隆 LeRobot 官方仓库并以可编辑模式安装：

```bash
#安装lerobot环境
conda create -y -n lerobot python=3.12
conda activate lerobot
python -m pip install --upgrade pip
conda install -c conda-forge "ffmpeg=6.1.1"

# 在准备安装 LeRobot 的目录中执行以下命令。
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

# Pipette 策略与通信脚本使用的额外依赖。
pip install transformers accelerate peft
pip install num2words
pip install pyzmq
pip install h5py
```

#### 安装 Pipette

```bash
# 在准备安装 Pipette 的目录中执行以下命令。
git clone https://github.com/hbhuiyou/Pipette.git
cd Pipette
```

执行后续命令时请保持终端位于 Pipette 仓库根目录，确保 `Scripts/...`、`Asset/...`、`datasets/...` 和 `models/...` 等路径能够统一解析。

Agent 启动 LeRobot 命令时还会清理 `PYTHONHOME` 和 `PYTHONPATH`，进一步避免环境冲突。

### 2. 配置本地路径

仓库中的任务预设使用 `Asset/Scene/lab_0.usd` 等相对于 Pipette 根目录的 USD 路径，运行时会自动从仓库根目录解析。

复制本地配置模板：

```bash
cp Scripts/Agent/local_config.example.env Scripts/Agent/local_config.env
```

填写实际路径：

```text
LEROBOT_PYTHON="/path/to/lerobot/python"
LEROBOT_MODEL_ROOT="models"
AGENT_ENV_TEMPLATE_USD="Asset/lab.usd"
AGENT_ASSET_DIR="Asset"
```

请勿将包含 API Key 或云服务密钥的 `Scripts/Agent/local_config.env` 提交到公开仓库。

## 快速开始

### 自然语言网页入口

```bash
python Scripts/Agent/web_agent.py
```

打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)，然后输入：

```text
我要采集数据
增强试管抓取数据
把 HDF5 转成 LeRobot
训练 SmolVLA
用 PI0 运行推理评估
```

Agent 的接口配置、网页控制和腾讯混元生 3D 资产生成功能请参阅 [`Scripts/Agent/README.md`](Scripts/Agent/README.md)。

## 数据流程

### 1. 采集示教数据

```bash
python Scripts/Data/Keyboard_collection.py \
  --task_id pick_up_the_tube \
  --num_demos 30 \
  --dataset_file ./datasets/pick_up_the_tube.hdf5
```

键盘快捷键：

- `R`：保存当前 demo
- `SPACE`：跳过当前 demo
- `P`：结束采集

### 2. 检查数据

```bash
python Scripts/Data/inspect_hdf5_dataset.py \
  --file ./datasets/pick_up_the_tube.hdf5 \
  --show-attrs
```

### 3. 生成仿真增强数据

```bash
python Scripts/Data/Generate_data.py \
  --task_id pick_up_the_tube \
  --dataset_file ./datasets/pick_up_the_tube.hdf5 \
  --output_file ./datasets/pick_up_the_tube_aug.hdf5 \
  --num_envs 3 \
  --light_intensity_scales 0.8 \
  --temporal_speed_scales 1.2 \
  --camera_jitter_count 5 \
  --include_original \
  --headless
```

增强过程不是直接修改离线图像，而是在仿真中重新执行轨迹并重新生成观测。当前支持：

- 光照强度扰动；
- 轨迹速度扰动和时间重采样；
- Top、Main 和 Wrist 相机位姿扰动；
- 有界关节动作噪声；
- 任务级成功筛选。

### 4. 转换为 LeRobot 数据集

在 LeRobot 环境中运行：

```bash
python Scripts/Data/hdf5_to_lerobot.py \
  --hdf5-path ./datasets/pick_up_the_tube_aug.hdf5 \
  --repo-id pick_up_the_tube \
  --output-dir /absolute/path/to/lerobot/pick_up_the_tube \
  --fps 10 \
  --frame-filter fresh \
  --stride 3
```

转换后的数据包含三路 RGB 图像、8 维机器人状态、8 维动作和语言指令。转换器会过滤视觉观测未及时更新的帧，以保持观测与动作的时序对齐。

## 训练

初次训练时，可以先在 LeRobot 环境中使用单个数据集运行 SmolVLA：

```bash
# 在Lerobot的目录下运行
lerobot-train \
  --dataset.repo_id=/path/to/lerobot/datasets/pick_tube \
  --policy.type=smolvla \
  --policy.repo_id=local/smolvla_pick_tube \
  --output_dir=/path/to/checkpoints/pick_tube_smolvla \
  --batch_size=8 \
  --steps=20000 \
  --wandb.enable=false
```

请将数据集和模型输出路径替换为实际的绝对路径。该示例仅训练 `pick_tube` 单个任务，适合在运行批量训练前验证数据集、显存和训练环境。

批量训练脚本支持 ACT、SmolVLA 和 PI0：

```bash
python Scripts/run_lerobot_batch_train.py \
  --model smolvla \
  --dataset-version aug \
  --dataset-root /path/to/lerobot/datasets \
  --output-root /path/to/checkpoints
```

正式训练前可使用 `--dry-run` 检查生成的命令：

```bash
python Scripts/run_lerobot_batch_train.py --model pi0 --dataset-version raw --dry-run
```

论文使用的训练配置如下：

| 策略 | Batch size | 训练步数 | 其他设置 |
|---|---:|---:|---|
| ACT | 32 | 15,000 | 默认精度 |
| SmolVLA | 8 | 20,000 | 默认精度 |
| PI0 | 4 | 20,000 | BF16、冻结视觉编码器、仅训练专家模块、梯度检查点 |

## 评估

评测系统由 LeRobot 策略服务端和 Isaac Lab 客户端组成，二者通过 ZMQ 通信。

在 LeRobot 环境中启动策略服务：

```bash
python Scripts/Server/server_brain.py \
  --policy-path /path/to/checkpoint \
  --policy-type smolvla \
  --bind tcp://127.0.0.1:5555 \
  --device cuda
```

启动对应的 Isaac Lab 客户端：

```bash
python Scripts/Client/inference_smolvla.py \
  --task-id pick_up_the_tube \
  --server-endpoint tcp://127.0.0.1:5555 \
  --episodes 100 \
  --output-json ./outputs/smolvla_pick_up_the_tube.json
```

当前提供以下客户端：

- `Scripts/Client/inference_act.py`
- `Scripts/Client/inference_smolvla.py`
- `Scripts/Client/inference_pi0.py`

每个 episode 会记录成功或失败、失败原因、运行时间、策略频率、控制频率和任务评估指标。

## 仓库结构

```text
.
|-- Asset/                  # 生成及用户提供的 USD 资产
|-- Scripts/
|   |-- Agent/             # 自然语言命令行与网页调度
|   |-- Client/            # Isaac Lab 策略评测客户端
|   |-- Data/              # 采集、回放、增强、转换和成功评估
|   |-- Server/            # 统一 LeRobot ZMQ 推理服务
|   `-- run_lerobot_batch_train.py
`-- README.md
```

## 当前限制

- 当前基准主要面向 Franka Panda 单臂任务。
- 尚未完成系统性的真实机器人验证和 Sim-to-Real 实验。
- 任务成功主要由针对不同任务编写的阈值评估器判定。
- 语言引导任务注册后，仍需人工确认 USD 路径、Prim 路径、相机配置和成功阈值。


## 致谢

Pipette 基于 [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 和 [LeRobot](https://github.com/huggingface/lerobot) 构建。
