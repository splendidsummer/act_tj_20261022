# 训练 Diffusion Policy

用 `conda` 环境 `act`（LeRobot 0.4.2，官方 `lerobot-train`，`--policy.type=diffusion`）。数据和已训完的 ACT 是同一份：LeRobot v3.0 的 MoveParcel。

README 里那条 Diffusion 命令不能直接用来训这份数据。正式训练已在 4090 上跑过，并在 10 月 12 日 20:14 从步数 40000 的检查点续上。

## 进度

10 月 12 日 08:57 停掉进程时，日志停在步数 59800，`loss` 为 0.006。当时还没写到步数 60000，磁盘上的检查点只有 20000 和 40000。续训只能从 40000 接着走，40000 到 59800 之间没保存的更新不保留。

截至 10 月 12 日 21:24，续训进程仍在跑。

| 项 | 值 |
|---|---|
| 步数 | 45800 / 200000 |
| `loss` | 0.007 |
| 学习率 | `8.8e-5` |
| 这一窗每步耗时 | 0.71 秒 |
| 检查点 | `checkpoints/020000`、`checkpoints/040000`，`last` 指向 40000 |
| 下一份检查点 | 步数 50000 |

配置里的 `1e-4` 是余弦调度的峰值，warmup 500 步，不是续训当下的学习率。调度器状态和步数 40000 的检查点一起加载。续训后第一条日志在 20:18，步数 40200，学习率 `9.1e-5`，与原来走到这一检查点时相同。到步数 45800 已降到 `8.8e-5`。

日志里的 `step` 会四舍五入成 K。精确步数用 `epch × 59494 / 32`。

## 开环动作误差

日志里的 `loss` 只是噪声预测误差。训练时模型看的是加了噪声的动作，要预测的是这份噪声（epsilon），`loss` 是预测噪声和真实噪声的均方误差。它不是解码出来的关节目标和演示动作之间的距离，所以 `loss` 降到 0.006 不能当成动作已经准了。

动作误差用开环单独算。评估方式和真机一致：每次取 2 帧观测，用 10 步 DDIM 采样出动作，执行其中 8 步，再采下一次。图像在评估时用中心裁剪。预测动作按 min/max 还原到训练前的单位，再和演示动作比。

脚本是 `outputs/eval/open_loop_diffusion.py`，结果写在 `outputs/eval/open_loop_diffusion.json`。`groups` 的单位是度或毫米。左夹爪在演示里恒为 0，它的误差会接近 0，不能当成夹爪学会了。

开环从步数 100000 开始，之后每 10000 步算一次：100000、110000，一直到 200000。100000 之前的检查点不算。每次只评 10 条回合，用种子 1000 从 100 条里随机抽出，各检查点用同一批回合。步数 100000 的检查点写下之后，训练改走 `outputs/train/train_diffusion_open_loop.py` 续上。它先算 100000 的动作误差，再接着训；以后每保存一份不早于 100000、且为 10000 倍数的检查点就再算一次。算失败只打印错误，训练继续。

## 数据

| 项 | 值 |
|---|---|
| 路径 | `/home/holdon/Projects/act_tj/lerobot_data/MoveParcel_20260928_1550` |
| `repo_id` | `local/MoveParcel_20260928_1550` |
| 版本 | LeRobot `v3.0` |
| 回合 / 帧 | 100 回合，59494 帧，30 Hz |
| 状态、动作 | 各 21 维，与 ACT 相同的顺序和单位 |
| 相机 | `observation.images.cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 图像 | 每路 `744×960`，AV1，解码后端 `torchcodec` |

## 和 README 的差别

README 里的 Diffusion 小节对不上这份数据，直接照抄会在开训前或第一步失败：

- 数据集路径写成了 `datasets/bimanual_pick`。实际目录是上面的 `lerobot_data/MoveParcel_20260928_1550`。
- 漏了 `--policy.push_to_hub=false`。LeRobot 0.4.2 里这项默认是 `true`，不关就会立刻要求 `policy.repo_id` 并退出。
- `--policy.crop_shape=null` 会把完整的 `744×960` 送进三路各自的 ResNet18。`batch_size=32` 时，前向已经占用约 20.8 GiB，24 GB 的 4090 在这一步显存不足。

LeRobot 0.4.2 的 Diffusion Policy 没有把图像缩小的参数。`--policy.crop_shape='[240,320]'` 是裁剪：训练时从每帧里随机裁一块高 240、宽 320 的区域，评估时改为中心裁剪。图像其余部分不进入网络。

## 训练

`gradient_accumulation_steps` 保持默认 1，所以有效批次就是 32。日志里应出现：

```text
Effective batch size: 32 x 1 x 1 = 32
```

预设优化器是 Adam，学习率 `1e-4`，余弦调度，warmup 500 步。状态和动作用 min/max 归一化，图像用 ImageNet 均值和标准差。`drop_n_last_frames` 默认是 7，与 `horizon=16`、`n_action_steps=8`、`n_obs_steps=2` 一致。

输出目录是新的 `outputs/train/diffusion_move_parcel`。不要复用 ACT 的 `outputs/train/act_move_parcel`。目录已经存在时，不带 `--resume=true` 会直接报错。

```bash
conda activate act
cd /home/holdon/Projects/act_tj

lerobot-train \
  --dataset.repo_id=local/MoveParcel_20260928_1550 \
  --dataset.root=/home/holdon/Projects/act_tj/lerobot_data/MoveParcel_20260928_1550 \
  --dataset.video_backend=torchcodec \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.crop_shape='[240,320]' \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --batch_size=32 \
  --steps=200000 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --output_dir=/home/holdon/Projects/act_tj/outputs/train/diffusion_move_parcel \
  --job_name=diffusion_move_parcel \
  --wandb.enable=false
```

关掉终端后还要继续跑时：

```bash
conda activate act
cd /home/holdon/Projects/act_tj

setsid nohup lerobot-train \
  --dataset.repo_id=local/MoveParcel_20260928_1550 \
  --dataset.root=/home/holdon/Projects/act_tj/lerobot_data/MoveParcel_20260928_1550 \
  --dataset.video_backend=torchcodec \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.crop_shape='[240,320]' \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --batch_size=32 \
  --steps=200000 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --output_dir=/home/holdon/Projects/act_tj/outputs/train/diffusion_move_parcel \
  --job_name=diffusion_move_parcel \
  --wandb.enable=false \
  > /home/holdon/Projects/act_tj/outputs/train/diffusion_move_parcel.log 2>&1 &
```

看进度：

```bash
tail -f /home/holdon/Projects/act_tj/outputs/train/diffusion_move_parcel.log
```

检查点每 10000 步写到 `outputs/train/diffusion_move_parcel/checkpoints/`。从步数 40000 续上后，下一份是 `checkpoints/050000`。稳定后每步大约 0.5 到 1 秒，10 万步量级要数小时，20 万步要一天多。
