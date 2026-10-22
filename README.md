# act_tj

用**官方 LeRobot ACT** 训练一个 21 自由度双臂躯干策略。只有当 ACT 把同一任务里两种同样合理的做法平均掉时，才使用**官方 Diffusion Policy**。

本项目不训练 π0、π0-FAST、SmolVLA 或 GR00T。

## 环境

Conda 环境名：`act`

```bash
conda activate act
```

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| LeRobot | 0.4.2（官方 `lerobot-train`，`--policy.type=act` / `--policy.type=diffusion`） |
| PyTorch | 2.7.1 + CUDA 12.6 轮子，给 RTX 4090 用 |

Hugging Face 没有 `lerobot` 0.3.4 这个发布版本。下面的命令对应 LeRobot 0.4.2，这个版本提供 `lerobot-train` 和 `--dataset.root`。

ACT 和 Diffusion Policy 是 LeRobot 自带的策略。这个环境不安装 π0、SmolVLA 或 GR00T 的额外依赖。

检查环境：

```bash
conda activate act
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; print('official ACT and Diffusion Policy ok')"
```

## 把数据集拷到这里

把一份 LeRobot 数据集放到：

```text
/home/holdon/Projects/act_tj/datasets/bimanual_pick/
```

拷完之后，这个目录应当是：

```text
datasets/bimanual_pick/
  meta/info.json
  meta/episodes.jsonl
  data/
  videos/
```

`lerobot-record` 把 `dataset.single_task` 存成回合标签。ACT 不读取这串文字。每个任务用一个固定字符串。

`meta/info.json` 里必须有这些特征：

- `fps`：30
- `observation.state` 形状为 `[21]`
- `action` 形状为 `[21]`
- 四路视频，尺寸都是 `[480, 640, 3]`：
  - `observation.images.left_wrist`
  - `observation.images.right_wrist`
  - `observation.images.top`（顶部深度相机的彩色图）
  - `observation.images.head`（一路头部鱼眼，去畸变后的针孔裁剪图）

第二路鱼眼和深度图留在磁盘上，不送进第一版策略。LeRobot 要求策略用到的每一路 `observation.images.*` 高和宽一致。

状态和动作使用同一顺序、同一单位。动作是绝对关节目标，不是增量。每个夹爪只存一次指令，不要把两根手指各存一维。

| 索引 | 关节 | 单位 |
|---:|---|---|
| 0–6 | 左臂，`Joint1_L` … `Joint7_L` | rad |
| 7 | 左夹爪，一个指令同时驱动 `Arm_L10_Joint` 和 `Arm_L11_Joint` | m |
| 8–14 | 右臂，`Joint1_R` … `Joint7_R` | rad |
| 15 | 右夹爪，一个指令同时驱动 `Arm_R10_Joint` 和 `Arm_R11_Joint` | m |
| 16 | 腰升降，`waist_jiont_1_prismatic` | m |
| 17 | 腰，`waist_joint_2` | rad |
| 18 | 腰，`waist_joint_3` | rad |
| 19 | 头，`head_joint_1` | rad |
| 20 | 头，`head_joint_2` | rad |

时刻 `t` 写入的动作，是当时发出的关节目标。状态是时刻 `t` 测到的关节位置。两者长度都是 21。

## 训练官方 ACT

在 RTX 4090 上，四路 480×640 相机、`batch_size=8`，大约占用 8–12 GB。10 万步通常要几小时。

```bash
conda activate act
cd /home/holdon/Projects/act_tj

lerobot-train \
  --dataset.repo_id=local/bimanual_pick \
  --dataset.root=/home/holdon/Projects/act_tj/datasets/bimanual_pick \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=20000 \
  --log_freq=200 \
  --num_workers=4 \
  --output_dir=/home/holdon/Projects/act_tj/outputs/train/act_bimanual_pick \
  --job_name=act_bimanual_pick \
  --wandb.enable=false
```

- `chunk_size=100` 在 30 Hz 下大约是 3.3 秒的未来关节目标。
- `n_action_steps=1` 加上 `temporal_ensemble_coeff=0.01` 表示每个控制周期都查询一次网络，并把重叠的动作段混合起来。这个 LeRobot 版本在 `n_action_steps` 大于 1 时不允许时间集成。
- 默认优化器是 AdamW，学习率 `1e-5`。保持不动。
- 视觉骨干是带 ImageNet 权重的 ResNet18。它不是语言模型。

检查点写在 `outputs/train/act_bimanual_pick`。如果到 10 万步损失还在明显下降，先再训 5 万步，先不要改机器人。

## 官方 Diffusion Policy，以后再用

只有当同一场景在数据里确实有两种成功行为（左手或右手、两条接近路线），而 ACT 把它们平均成一段碰不到物体的动作时，才用这一节。

这一轮把四路训练图像缩到 240×320，让 24 GB 的 4090 放得下。21 维关节向量保持不变。

```bash
conda activate act
cd /home/holdon/Projects/act_tj

lerobot-train \
  --dataset.repo_id=local/bimanual_pick \
  --dataset.root=/home/holdon/Projects/act_tj/datasets/bimanual_pick \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.crop_shape=null \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --batch_size=16 \
  --steps=200000 \
  --save_freq=20000 \
  --num_workers=4 \
  --output_dir=/home/holdon/Projects/act_tj/outputs/train/diffusion_bimanual_pick \
  --job_name=diffusion_bimanual_pick \
  --wandb.enable=false
```

Diffusion Policy 的默认配置是给 PushT 用的（`crop_shape` 为 84×84、编码器共享、100 步 DDPM）。上面的参数把默认值换成真实相机配置。推理用 10 步 DDIM，这样 4090 仍能闭环。显存大约 16–22 GB，通常要过夜。

## 本项目不包含

- 不训练 π0、π0-FAST、SmolVLA、GR00T。
- 第一版策略不使用第二路头部鱼眼，也不使用深度通道。
- 第一个任务没有记下成功率之前，不开始第二个任务。
- ACT 训练用的图像不超过 480×640。
