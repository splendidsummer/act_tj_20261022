# 部署推理服务

推理代码在 `/home/holdon/Projects/act_tj_20261022`。服务端在 GPU 机器上，客户端和下面的 ROS 服务在机器人 `nvidia@6.6.7.100` 上。预览图没有拷过去。

| 路径 | 作用 |
|---|---|
| `serving/` | 服务端。`quad_image.py` 把两路腕部图底部 48 行涂黑，头顶相机不动 |
| `client/gento_lerobot_client.py` | 机器人客户端 |
| `scripts/serve_lerobot_policy.sh` | 启动脚本。进入仓库后执行本目录的 `.venv/bin/python` |
| `docs/ROS2_TOPICS.md` | 客户端用到的话题、21 维状态和 mux 源 |

## 环境和端口

虚拟环境在本仓库：`/home/holdon/Projects/act_tj_20261022/.venv`。它是从原来的 `/home/holdon/Projects/act_tj/.venv` 克隆的，那份已经删除。Python 3.10.22，PyTorch `2.7.1+cu126`。

一次只开一个服务，默认端口 8000。换模型先停掉当前进程。

## 检查点

| 策略 | 训练名 | 步数 | 返回 | 路径 |
|---|---|---:|---|---|
| ACT | `act_flip_and_remove` | 100000 | 30 步动作 | `/home/holdon/Projects/act_tj_20261022/outputs/train/act_flip_and_remove/checkpoints/100000/pretrained_model` |
| Diffusion | `diffusion_move_parcel` | 200000 | `--chunk-size 30` 在服务里压成 8 步 | `/home/holdon/Projects/act_tj_20261022/outputs/train/diffusion_move_parcel/checkpoints/200000/pretrained_model` |

`CKPT` 指到 `pretrained_model` 这一层，不是上面的步数目录。

## ACT

在 GPU 机器上。`act_flip_and_remove` 的 100000 步，返回 30 步动作：

```bash
cd /home/holdon/Projects/act_tj_20261022
CKPT=/home/holdon/Projects/act_tj_20261022/outputs/train/act_flip_and_remove/checkpoints/100000/pretrained_model \
  bash scripts/serve_lerobot_policy.sh
```

## Diffusion

在 GPU 机器上。`diffusion_move_parcel` 的 200000 步。`--chunk-size 30` 会在服务里被压成 8 步：

```bash
cd /home/holdon/Projects/act_tj_20261022
CKPT=/home/holdon/Projects/act_tj_20261022/outputs/train/diffusion_move_parcel/checkpoints/200000/pretrained_model \
  bash scripts/serve_lerobot_policy.sh
```

## 起来之后

日志里应看到 `Loaded ... (act)` 或 `Loaded ... (diffusion)`，以及 `websocket server on ws://0.0.0.0:8000`。Diffusion 还会打出实际返回 8 步。换模型要先停掉当前进程。

## 机器人上，先切模式

策略服务不起机器人。在机器人上加载 APEX 环境后再调服务。`APEX_ROS_NAMESPACE=tj`，名字要带 `/tj` 前缀；写成 `/control/...` 会找不到。

2026-10-22 按这个顺序调用过，三次都返回成功：

```bash
source /etc/apex/apex_ros_env.sh
ros2 service call /tj/control/clear_fault std_srvs/srv/Trigger "{}"
ros2 service call /tj/control/set_ready   std_srvs/srv/Trigger "{}"
ros2 service call /tj/control/set_mode    marvin_msgs/srv/Int "{data: 3}"
```

| 顺序 | 服务 | 这次的返回 |
|---:|---|---|
| 1 | `/tj/control/clear_fault` | `success=True`，`Robot faults cleared successfully.` |
| 2 | `/tj/control/set_ready` | `success=True`，`Robot is ready.` |
| 3 | `/tj/control/set_mode`，`data: 3` | `success=True`，`error_code=-1`，消息为空 |

`set_mode` 切的是机器人运行模式。客户端进循环时还会自己调用 `/tj/control/set_input`，`data=3`，那是 mux 的 user 源。两个服务不是同一个。

关节指令 mux 随遥操作 launch 起来。它不在时，`/tj/control/user/joint_cmd_*` 不在图里，客户端发出去的关节指令没有接收者。

## 机器人客户端

模式切好、GPU 上的服务已经在听 8000 之后，在机器人上启动。默认连 `6.6.7.134:8000`，`--replan-steps` 不写就用服务端的 `chunk_size`（ACT 为 30）。

```bash
source /etc/apex/apex_ros_env.sh
python3 client/gento_lerobot_client.py \
  --policy-host 6.6.7.134 \
  --policy-port 8000
```
