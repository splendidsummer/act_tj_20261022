# ACT WebSocket 服务（真机 RTC）

GPU 上加载 ACT，按 OpenPI 的 WebSocket 协议把**一段绝对关节动作**发给机器人。机器人端仍用 `lerobot_rtc_inference` 上现成的 RTC 客户端做预取和 chunk 交接。

代码在 `/home/holdon/Projects/openpi_tj`，分支 `lerobot_rtc_inference`：

| 文件 | 作用 |
|---|---|
| `scripts/serve_policy_act_gento_ws.py` | 服务进程 |
| `openpi_gento_act_serve.sh` | 在 `act` 环境里启动服务 |
| `openpi_gento_act_rtc_client.sh` | 机器人上启动 RTC 客户端 |

`docs/DEPLOY.md` 里的 `serving/serve_policy.py` 是另一条路径：每个控制周期返回一个 21 维动作，并在服务端做 temporal ensemble。本文这个服务一次返回一个 chunk，不做 temporal ensemble。

## 启动

在 GPU 机器上，默认加载第 100000 步，监听 `0.0.0.0:8000`，返回模型 100 步预测里的前 30 步。

```bash
cd /home/holdon/Projects/openpi_tj
bash openpi_gento_act_serve.sh
```

换检查点和端口：

```bash
CHECKPOINT_DIR=/home/holdon/Projects/act_tj/outputs/train/act_move_parcel/checkpoints/080000/pretrained_model \
  CHUNK_SIZE=30 \
  bash openpi_gento_act_serve.sh --port 8000
```

| 变量 / 参数 | 默认 | 含义 |
|---|---|---|
| `CHECKPOINT_DIR` / `--checkpoint-dir` | `.../checkpoints/100000/pretrained_model` | 含 `model.safetensors` 和 preprocessor 的目录 |
| `CHUNK_SIZE` / `--chunk-size` | `30` | 返回的时间步数，范围 `1..100` |
| `SWAP_WAIST=0` / `--no-swap-waist` | 交换开着 | 关掉第 17、18 维的腰轴交换 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 端口 |
| `--no-warmup` | 启动时先跑一帧 | 跳过这帧 |

环境是 conda 的 `act`，需要 CUDA。`openpi_gento_act_serve.sh` 会自己 `conda activate act`。OpenPI 的 `.venv` 里没有这份 LeRobot ACT，不要用 `uv run` 起这个服务。

`GET http://主机:8000/healthz` 返回 `OK`。

机器人端，控制层就绪之后：

```bash
cd /home/holdon/Projects/openpi_tj
POLICY_HOST=<GPU 机器 IP> bash openpi_gento_act_rtc_client.sh
```

这个脚本就是原来的 `openpi_gento_rtc_client.sh`，并加上 `--no-startup-pose`。笔任务的预设姿态不在这批演示的腰部范围内。

## 连接上之后

协议和 OpenPI 的 `WebsocketPolicyServer` 相同，消息用 msgpack，可以带 NumPy 数组。压缩关掉，单条消息没有大小上限。

1. 客户端连上 `ws://主机:8000`。
2. 服务先发一条元数据。
3. 之后客户端每发一帧观测，服务回一个字典。
4. 出错时服务先发一段文本 traceback，再关掉连接。

元数据字段：

| 键 | 值 |
|---|---|
| `server` | `serve_policy_act_gento_ws` |
| `policy_type` | `act` |
| `checkpoint_dir` | 本次加载的目录 |
| `state_dim` / `action_dim` | `21` |
| `chunk_size` | 实际返回的步数，默认 `30` |
| `model_chunk_size` | `100`，模型一次预测的长度 |
| `action_rate` | `30.0` |
| `output_action_schema` | 21 个关节名，顺序和训练数据一致 |
| `action_mode` | `absolute` |
| `image_height` / `image_width` | `744` / `960` |
| `cameras` | `cam_high`，`cam_left_wrist`，`cam_right_wrist` |
| `swap_waist` | 是否交换腰的两维 |

返回字典：

| 键 | 含义 |
|---|---|
| `actions` | `float32`，形状 `(chunk_size, 21)`，已经反归一化的绝对关节目标 |
| `image_decode_ms` | 解码四宫格花的毫秒 |
| `server_timing.infer_ms` | 这一次 `infer` 的毫秒，含解码 |
| `server_timing.prev_total_ms` | 上一帧从收到观测到发完回复的毫秒；第一帧没有这项 |

第 100000 步在本机做过一帧预热：三路全零图、21 维全零状态，返回的 `actions` 形状是 `(30, 21)`，数值有限，量级是弧度。左夹爪那一维是 `0`，因为训练动作里它恒为 0。

## 观测

真机客户端 `vla_client/openpi_client_gento.py` 发的是压缩四宫格，不是三张已经拆开的图。

```python
{
    "compressed_image": {
        "packets": [b"..."],          # JPEG / PNG 用最后一包；H264 按顺序喂给解码器
        "format": "h264",             # 或 jpeg / png
        "quad_width": 1920,
        "quad_height": 1488,
        "image_size": 224,            # 这个服务不使用该字段
    },
    "state": state21,                 # float32, (21,)
    "prompt": "replace the object",   # ACT 不读这串文字
}
```

四宫格拆法：左上 `cam_high`，左下 `cam_left_wrist`，右下 `cam_right_wrist`。每一路收成训练时的 **960×744** RGB。块的尺寸如果不是这个，会用面积插值缩放到 960×744。腕部底部不涂黑。

`state` 必须是 21 维。顺序与 `docs/DEPLOY.md` 的关节表相同。服务内部把它和三路图变成：

- `observation.state`：`float32`，形状 `(21,)`
- `observation.images.cam_high` 等：`float32`，形状 `(3, 744, 960)`，范围 `0–1`

然后走检查点里的 preprocessor（含 ImageNet 均值和标准差）。模型用 `predict_action_chunk`，VAE latent 固定为 0，得到归一化的 100 步。再用 postprocessor 里的 `action.mean` 和 `action.std` 乘回去，截成 `--chunk-size` 步发出去。

直接传入已经拆好的图也可以，预热走的就是这条：

```python
{
    "images": {
        "cam_high": high,              # uint8 或 float, HWC 或 CHW
        "cam_left_wrist": left,
        "cam_right_wrist": right,
    },
    "state": state21,
}
```

## 腰的两维

数据集第 17 维是 `waist_yaw`（大约 ±0.44 rad），第 18 维是 `waist_pitch`（大约 −0.73 到 0）。RTC 客户端从 `/joint_states` 里按 body 的原顺序填进这两维，那个顺序是先 pitch 再 yaw。

服务默认在送进模型之前交换第 17 和 18 维，动作出来之后再交换回去，这样客户端写回 `JointcmdBody` 时轴还是原来的。第一次动作前看客户端日志里的 `state[21]`：摆成演示附近的站姿时，第 17 个数应接近 0，第 18 个数应为负数。若这两个轴反了，用 `SWAP_WAIST=0` 重启服务。

## 和客户端 RTC 怎么接上

服务只负责生成 chunk。队列、推理延迟补偿、新旧 chunk 融合都在客户端。

默认配合是：服务返回 30 个 policy step，客户端 `--action-rate 30`、`--control-rate 200`，队列剩余 20 个 policy step 时预取下一块，交接处用 `exp` 融合 10 个 policy step。同一次比较里不要中途改这些数。

prompt 被忽略。一条连接不会在连上时调用 `policy.reset()`。这个服务没有按时间集成去混历史预测，所以也不需要 `{"reset": true}`。
