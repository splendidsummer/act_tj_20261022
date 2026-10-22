# 续训 ACT

当前这次训练的输出目录是 `/home/holdon/Projects/act_tj/outputs/train/act_move_parcel`。检查点每 20000 步保存一次。停掉进程后，只能从最后一份完整检查点接着训，停掉当下还没保存的步数不会保留。

## 检查点

| 路径 | 内容 |
|---|---|
| `outputs/train/act_move_parcel/checkpoints/020000` | 第 20000 步 |
| `outputs/train/act_move_parcel/checkpoints/040000` | 第 40000 步 |
| `outputs/train/act_move_parcel/checkpoints/last` | 指向最新一份，目前是 `040000` |

每一份里面有：

- `pretrained_model/model.safetensors`：策略权重
- `pretrained_model/train_config.json`：这次训练的全部配置
- `training_state/optimizer_state.safetensors`：AdamW 状态
- `training_state/training_step.json`：已完成的步数
- `training_state/rng_state.safetensors`：随机数状态

续训前先看最新一份停在哪一步：

```bash
cat /home/holdon/Projects/act_tj/outputs/train/act_move_parcel/checkpoints/last/training_state/training_step.json
```

## 学习率

学习率固定为 **1e-5**，没有 warmup，也没有衰减。`train_config.json` 里 `scheduler` 是 `null`。主干网络和 ResNet18 backbone 都是这个学习率，权重衰减 `1e-4`。

续训会加载优化器状态，学习率仍然是 `1e-5`，不会回到初始状态重来。

## 接着训

用最新检查点，从已完成的步数继续跑到 10 万步。配置里已经写好数据集、`batch_size=16`、`gradient_accumulation_steps=2`（有效批次 32）、`steps=100000` 和原来的输出目录。

```bash
conda activate act
cd /home/holdon/Projects/act_tj

setsid nohup lerobot-train \
  --resume=true \
  --config_path=/home/holdon/Projects/act_tj/outputs/train/act_move_parcel/checkpoints/last/pretrained_model/train_config.json \
  > /home/holdon/Projects/act_tj/outputs/train/act_move_parcel.log 2>&1 &
```

`setsid nohup` 是为了关掉终端后训练还在。日志追加在原来的 `outputs/train/act_move_parcel.log`。

看进度：

```bash
tail -f /home/holdon/Projects/act_tj/outputs/train/act_move_parcel.log
```

日志里的 `step` 会被四舍五入成 K。精确步数用 `epch` 估算：`步数 ≈ epch × 59494 / 32`。`loss` 是归一化动作的 L1 加上 `10 × KL`。

下一次保存仍按 20000 步对齐，例如从 40000 续上后，下一份是 `checkpoints/060000`。

不要再跑 README 里那条全新的 `lerobot-train`。输出目录已经存在，不带 `--resume=true` 会直接报错，避免把这次训练覆盖掉。
