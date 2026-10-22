# 推理客户端的 ROS 2 话题

服务端 `serving/` 不连接 ROS。话题都在机器人上的 `client/gento_lerobot_client.py` 里。循环 500 Hz，每次 `spin_once` 收最新消息；动作队列空了才组观测并向策略要下一段。

2026-10-22 在 `nvidia@6.6.7.100` 上核对过。先加载 APEX 环境，再调 `ros2`：

```bash
source /etc/apex/apex_ros_env.sh
```

加载后是 `ROS_DOMAIN_ID=10`、`rmw_cyclonedds_cpp`、`CYCLONEDDS_URI=file:///etc/apex/cyclonedds.xml`。`APEX_ROS_NAMESPACE=tj`，所以服务和话题都带 `/tj` 前缀。写成 `/control/clear_fault` 会找不到；实际名字是 `/tj/control/clear_fault`。

下面这些名字和类型仍然对得上，客户端不用为这次遥操作升级改话题。

关节指令要经过 `/tj/joint_cmd_mux`。这个节点随遥操作 launch 起来；它不在时，`/tj/control/user/joint_cmd_*` 不会出现在图里。

## 推理前

GPU 上的策略服务不起机器人。在机器人上、客户端之前，按这个顺序调三次。2026-10-22 16:35 这三次都返回了成功。

```bash
source /etc/apex/apex_ros_env.sh
ros2 service call /tj/control/clear_fault std_srvs/srv/Trigger "{}"
ros2 service call /tj/control/set_ready   std_srvs/srv/Trigger "{}"
ros2 service call /tj/control/set_mode    marvin_msgs/srv/Int "{data: 3}"
```

| 顺序 | 服务 | 类型 | 这次的返回 |
|---:|---|---|---|
| 1 | `/tj/control/clear_fault` | `std_srvs/srv/Trigger` | `success=True`，`Robot faults cleared successfully.` |
| 2 | `/tj/control/set_ready` | `std_srvs/srv/Trigger` | `success=True`，`Robot is ready.` |
| 3 | `/tj/control/set_mode` | `marvin_msgs/srv/Int` | `data: 3`，`success=True`，`error_code=-1`，消息为空 |

`set_mode` 是机器人运行模式。它和下面的 `set_input` 不是同一个服务。客户端进循环时仍会自己调用 `/tj/control/set_input`，`data=3`。

## 订阅

开环只等 `/tj/joint_states` 和图像。两路夹爪反馈缺失时，对应维度记 0，循环照样走。

| 话题 | 类型 | 发布者 | 核对时的频率 | 客户端用法 |
|---|---|---|---|---|
| `/tj/joint_states` | `sensor_msgs/msg/JointState` | `/tj/marvin_robot_node` | 约 90–100 Hz | 19 维位置，拆成臂、升降、腰、头 |
| `/tj/info/gripper_feedback_L` | `std_msgs/msg/Float32MultiArray` | `/tj/dm_gripper_motor_node` | 约 200 Hz | `data[0]` 裁到 0–1，状态第 7 维 |
| `/tj/info/gripper_feedback_R` | `std_msgs/msg/Float32MultiArray` | `/tj/dm_gripper_motor_node` | 约 200 Hz | `data[0]` 裁到 0–1，状态第 15 维 |
| `/quad_tile/compressed_undistorted` | `sensor_msgs/msg/CompressedImage` | `/quad_csi_quickview` | 约 30 Hz | 默认图像话题，`--image-topic` 可改。重规划时把积下的压缩包发给服务器 |

图像格式是 `h264`，一帧大约 40 KB。发布端 QoS 是 `BEST_EFFORT`、`KEEP_LAST 1`。客户端用 `qos_profile_sensor_data`。

夹爪反馈现在是 5 个数。客户端只取第一位。核对时左约为 `[0.011, -0.007, -0.037, 32, 30]`，右约为 `[-0.004, -0.007, 0.002, 30, 29]`。

`/tj/joint_states` 的 19 个名字：

```text
Joint1_L … Joint7_L
Joint1_R … Joint7_R
waist_joint_1
waist_joint_2
waist_joint_3
Head_J1_Joint
Head_J2_Joint
```

填进 21 维状态时，腰的两维对调。`/tj/joint_states` 里 body 是先 `waist_joint_2` 再 `waist_joint_3`；策略状态是先 `waist_yaw` 再 `waist_pitch`。

| 状态下标 | 含义 | 来源 |
|---:|---|---|
| 0–6 | 左臂 | `position[0:7]` |
| 7 | 左夹爪 | 左夹爪反馈 `data[0]` |
| 8–14 | 右臂 | `position[7:14]` |
| 15 | 右夹爪 | 右夹爪反馈 `data[0]` |
| 16 | 升降 | `position[14]`，`waist_joint_1` |
| 17 | `waist_yaw` | `position[16]`，`waist_joint_3` |
| 18 | `waist_pitch` | `position[15]`，`waist_joint_2` |
| 19 | `head_yaw` | `position[17]`，`Head_J1_Joint` |
| 20 | `head_pitch` | `position[18]`，`Head_J2_Joint` |

## 发布

`publish_action()` 每个 500 Hz 节拍发下面 6 个话题。等推理结果时发上一条指令，结果回来后发插值后的下一步。`--disable-action-post` 时不发。

| 话题 | 类型 | 内容 |
|---|---|---|
| `/tj/control/user/joint_cmd_A` | `marvin_msgs/msg/JointcmdArm` | 左臂 7 关节 |
| `/tj/control/user/joint_cmd_B` | `marvin_msgs/msg/JointcmdArm` | 右臂 7 关节 |
| `/tj/control/user/joint_cmd_body` | `marvin_msgs/msg/JointcmdBody` | `[升降, waist_pitch, waist_yaw, 0, 0, 0]` |
| `/tj/control/user/joint_cmd_head` | `marvin_msgs/msg/JointcmdHead` | `[head_yaw, head_pitch, 0]` |
| `/tj/control/gripperValueL` | `std_msgs/msg/Float32` | 左夹爪，订阅者含 `/tj/dm_gripper_motor_node` |
| `/tj/control/gripperValueR` | `std_msgs/msg/Float32` | 右夹爪，订阅者含 `/tj/dm_gripper_motor_node` |

四路 `joint_cmd_*` 由 `/tj/joint_cmd_mux` 订阅，QoS 是 `BEST_EFFORT`、`KEEP_LAST 1`。客户端用默认可靠 QoS、队列深度 10 发布，和这个订阅兼容。mux 再发布到 `/tj/control/joint_cmd_A|B|body|head`，由 `/tj/marvin_robot_node` 接收。

mux 的源：

| `active_source` | 名字 | 前缀 |
|---:|---|---|
| 0 | teleop | `/tj/control/qp_controller` |
| 1 | planner | `/tj/control/plan` |
| 2 | user | `/tj/control/user` |
| 3 | replay | `/tj/control/replay` |

推理客户端走的是 `user`。另外还有 `/tj/control/replay` 等前缀，那些不是这条客户端的发布话题。

## 服务

推理前的 `/tj/control/clear_fault`、`/tj/control/set_ready`、`/tj/control/set_mode` 见上面一节。`set_mode` 的 `data: 3` 只切机器人运行模式。

`/tj/control/set_input`，类型 `marvin_msgs/srv/Int`，由 `/tj/joint_cmd_mux` 提供。请求字段是 `int64 data`。

兼容映射是 `源编号 = data - 1`：

| `data` | 切到 |
|---:|---|
| 0 | 空闲（源 `-1`） |
| 1 | teleop |
| 2 | planner |
| 3 | user |
| 4 | replay |

客户端进循环前调用 `data=3`，退出时调用 `data=0`。直接选源用 `/tj/control/joint_cmd_mux/select`，那里的 `data` 就是上表的源编号，`2` 才是 user。

当前选中的源在 `/tj/info/joint_cmd_mux/active_source`（`std_msgs/msg/Int32`）。`/tj/control/input_mode` 是同一套兼容编号，teleop 时为 `1`。
