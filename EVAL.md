# ACT 实机评估

检查点是同一次训练的四份权重，任务是数据集里的 `remove the object`。动作是 21 维绝对关节目标，频率 30 Hz。没有仿真器，所以评估分成两段：先在演示数据上算开环误差，再在真机上用 RTC 闭环。

| 步数 | 目录 |
|---|---|
| 40000 | `outputs/train/act_move_parcel/checkpoints/040000/pretrained_model` |
| 60000 | `outputs/train/act_move_parcel/checkpoints/060000/pretrained_model` |
| 80000 | `outputs/train/act_move_parcel/checkpoints/080000/pretrained_model` |
| 100000 | `outputs/train/act_move_parcel/checkpoints/100000/pretrained_model` |

训练 loss 从 40000 步的 0.056 降到 100000 步的 0.037。开环误差和真机成功率不必按这个顺序单调变好，四份都要测。

## 1. 开环

脚本是 `outputs/eval/open_loop_act.py`。它对 100 个回合的每一帧做和真机相同的前向：VAE latent 固定为 0，`temporal_ensemble_coeff=0.01`，每帧只取混合后的一个动作，再和演示动作比。

这一段用的是 ACT 自己的时间集成，不是第 2 节的 RTC。它回答的是“模型有没有记住演示”，不是“机器人能不能做成”。

```bash
conda activate act
cd /home/holdon/Projects/act_tj
python outputs/eval/open_loop_act.py --steps 40000 60000 80000 100000
```

结果写在 `outputs/eval/open_loop_act.json`。`groups` 里的单位已经换成度或毫米。左夹爪动作在全部演示里都是 0，所以它的误差会是 0，不能当成夹爪学会了。

40000 步已经算完，59494 帧，归一化动作的平均绝对误差是 **0.060**。分组误差：

| 部位 | 平均绝对误差 |
|---|---|
| 左臂 | 0.64° |
| 右臂 | 1.06° |
| 右夹爪 | 16.3 mm |
| 腰升降 | 0.81 mm |
| 腰 yaw / pitch | 0.57° |
| 头 | 0.30° |

60000、80000、100000 用同一脚本接着算。四份都出来之后，看右臂和右夹爪：这两项在 40000 步最大。若 100000 步的右臂仍大于约 1°，或右夹爪仍大于约 15 mm，真机上要预期抓取点偏。

## 2. 真机 RTC

Pi0 的服务端会把图缩成 224，并把大部分关节当成增量。这份 ACT 不能走那条服务。需要的是一个仍使用现有 RTC 客户端的 ACT 服务：客户端负责异步预取和 chunk 交接，服务端返回绝对关节目标。

代码在 `openpi_tj` 的 `lerobot_rtc_inference` 分支：

| 文件 | 作用 |
|---|---|
| `scripts/serve_policy_act_gento_ws.py` | 加载 ACT，解码四宫格，返回绝对动作 |
| `openpi_gento_act_serve.sh` | GPU 机器上启动服务，使用 `act` 环境 |
| `openpi_gento_act_rtc_client.sh` | 机器人上启动现有 RTC 客户端，并关掉笔任务的起始姿态 |

服务端保持训练时的图像：四宫格拆成三路 **960×744**，不做 Pi0 的腕部底部涂黑。模型一次预测 100 步，默认只把前 **30** 步发给客户端。客户端仍是 30 Hz 动作、200 Hz 插值，队列剩 20 个 policy step 时预取，交接融合 10 步。

腰的顺序和客户端相反。数据集第 17 维是 yaw（约 ±0.44 rad），第 18 维是 pitch（约 −0.73 到 0）。ROS 的 body 两关节是先 pitch 再 yaw，现有客户端按这个顺序原样填进第 17、18 维。服务端默认对这两维做一次交换，动作返回前再交换回去。第一次动之前用下面的检查确认；若腰的两个轴反了，用 `SWAP_WAIST=0` 重开服务。

### 2.1 启动

GPU 机器：

```bash
cd /home/holdon/Projects/openpi_tj
bash openpi_gento_act_serve.sh
```

换检查点只改目录，客户端不用改：

```bash
CHECKPOINT_DIR=/home/holdon/Projects/act_tj/outputs/train/act_move_parcel/checkpoints/080000/pretrained_model \
  bash openpi_gento_act_serve.sh
```

机器人上，控制层就绪后：

```bash
cd /home/holdon/Projects/openpi_tj
source /etc/apex/apex_ros_env.sh
ros2 service call /control/clear_fault std_srvs/srv/Trigger "{}"
ros2 service call /control/set_ready   std_srvs/srv/Trigger "{}"
ros2 service call /control/set_mode    marvin_msgs/srv/Int "{data: 3}"

POLICY_HOST=<GPU 机器 IP> bash openpi_gento_act_rtc_client.sh
```

不要用 `openpi_gento_serve.sh`。不要打开 `--startup-pose`：那个预设姿态的腰是 0，不在这批演示的 pitch 范围内。把机器人先摆到一条演示开始时的姿态，再开客户端。

ACT 不读 prompt。左夹爪动作在训练里恒为 0，真机上左夹爪会保持关闭，这是数据决定的。

### 2.2 第一次动作前

看客户端日志里的 `state[21]`。摆一个和演示相近的站姿时，第 17 个数应接近 0 且落在 ±0.44 里，第 18 个数应为负数、大约 −0.3。若这两个数对调了，停掉，用 `SWAP_WAIST=0` 重启服务再看一次。

确认图像话题和录制时一致。客户端脚本默认 `/quad_tile/compressed_undistorted`。若演示来自未去畸变的 `/quad_tile/compressed`，启动时加上 `--image-topic /quad_tile/compressed`。

急停放在手边。软件限幅已经开着：腕关节每拍最多 0.004 rad，腰每拍最多 0.00075 rad。

### 2.3 每个检查点怎么打分

每个检查点做 **10** 次。场景、物体和起始摆放跟录制时一致，次数之间把物体放回同一位置。一次尝试从客户端开始发策略命令算起，到物体放到目标处、明显失败、或 60 秒，以先到者为准。

| 结果 | 标准 |
|---|---|
| 成功 | 物体离开原位，并放到演示里的目标处 |
| 半成功 | 抓住或移开了，但没放到目标处 |
| 失败 | 没形成有效接触，或者中途不得不按急停 |

先用 **100000** 做 2 次烟测。腰、腕没有来回抖，命令也没有跳变，再把这个检查点补满 10 次，然后按 80000、60000、40000 重复。换权重时只重启服务端。

每次记下：检查点、成功/半成功/失败、大约用时、客户端日志里的 `server_infer` 毫秒。10 次里的成功数就是这个检查点的结果。

若动作跟得上但交接发颤，把客户端的 `RTC_EXECUTION_HORIZON` 调到 12。若反应慢，把服务端 `CHUNK_SIZE` 降到 20，同时把 `RTC_QUEUE_THRESHOLD` 降到 12。不要在同一次比较里混用不同的 RTC 参数。

## 3. 不在这次做的

- 不把 ACT 接进 Pi0 的增量动作服务。
- 不在服务端再跑一遍 temporal ensemble。开环用它，真机用客户端 RTC，两套平滑不要叠在一起。
- 不在这四份权重的真机结果出来之前改网络或再训。
