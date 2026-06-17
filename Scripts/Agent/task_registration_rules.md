# Task Registration Rules

这个文件用于约束 Agent 的“注册任务”能力。注册任务时采用填空方式，用户只需要输入 USD 场景路径和任务描述；`task_id`、标准化 `description`、`language_instruction` 由大模型生成，Agent 校验后再写入 `Data/task_registry.py`。

## 用户输入

- `usd_path`
  - 必须是已经搭建好的 USD 场景路径。
  - 后缀必须是 `.usd`、`.usda` 或 `.usdc`。
- `任务描述`
  - 用户用自然语言描述任务即可。
  - 示例：`让机械臂拿起蓝色试管`、`让机械臂打开离心机盖子`。

## 大模型生成

- `task_id`
  - 由任务描述生成。
  - 必须是英文小写 snake_case。
  - 只能使用小写字母、数字、下划线。
  - 必须以小写字母开头。
  - 不能和现有任务重复；如重复，Agent 自动追加数字后缀。
- `description`
  - 由任务描述标准化生成。
  - 推荐格式：`Keyboard teleoperation data collection for xxx task.`
- `language_instruction`
  - 由任务描述生成。
  - 用英文小写 snake_case 表达机器人要完成什么。
  - 示例：`open_the_box`、`pick_up_the_blue_tube`。

## 自动生成

- `dataset_file`
  - 不再询问用户。
  - 固定生成：`./datasets/{task_id}.hdf5`。
  - 这与现有默认任务的保存路径格式保持一致。

## 固定模板

注册任务时先添加任务常量：

```python
OPEN_THE_BOX_TASK_ID = "open_the_box"
```

然后只能向 `TASK_PRESETS` 添加以下形式的条目：

```python
OPEN_THE_BOX_TASK_ID: make_task(
    task_id=OPEN_THE_BOX_TASK_ID,
    description="Keyboard teleoperation data collection for xxx task.",
    usd_path="/path/to/scene.usd",
    dataset_file="./datasets/open_the_box.hdf5",
    language_instruction="open_the_box",
),
```

## 禁止事项

- 不允许修改 `TaskPreset`、`TaskCameraSpec`、`TubeEvalConfig` 的字段定义。
- 不允许修改已有任务的内容，除非用户明确要求。
- 不允许修改默认机器人、相机、控制频率、成功判定参数。
- 不允许把任意 Python 表达式写入字段值，所有字段都必须作为普通字符串写入。
- 不允许覆盖已有 `task_id`。

## 后续高级配置

如果新任务需要专门的成功判定、物体 prim path 或相机位姿，应在注册完成后单独让用户确认，再对 `task_registry.py` 做人工可审查的小范围修改。
