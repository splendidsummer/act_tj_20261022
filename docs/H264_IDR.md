# 数相机流里的 IDR 和 P 帧

IDR 和 P 帧的比例由相机编码器决定，这个仓库和 ACT 都不设置它。客户端只在 IDR 出现时清空积压的包，然后从这个 IDR 重新开始攒。要知道这台机器人实际是几个 IDR、几个 P 帧，就在相机话题上数。

话题是 `sensor_msgs/msg/CompressedImage`。RTC 客户端默认订的是：

```text
/quad_tile/compressed_undistorted
```

未去畸变的流是 `/quad_tile/compressed`。两条都要用同一种数法；推理时客户端订哪条，就数哪条。

一个包里只要有一个 NAL 类型为 5 的单元，就是 IDR。`00 00 01` 后面那个字节的低 5 位等于 5。这和 `openpi_client_gento.py` 里的 `h264_contains_idr` 相同。IDR 之后、下一个 IDR 之前的包记成 P 帧。

## 在机器人上数

先确认话题在发：

```bash
source /etc/apex/apex_ros_env.sh
ros2 topic hz /quad_tile/compressed_undistorted
ros2 topic echo /quad_tile/compressed_undistorted --once --field format
```

`format` 里应能看到 `h264`。然后数 10 秒。把话题换成实际在用的那条即可。

```bash
source /etc/apex/apex_ros_env.sh
python3 - << 'PY'
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

TOPIC = "/quad_tile/compressed_undistorted"
SECONDS = 10.0

def is_idr(payload: bytes) -> bool:
    index = payload.find(b"\x00\x00\x01")
    while index != -1 and index + 3 < len(payload):
        if payload[index + 3] & 0x1F == 5:
            return True
        index = payload.find(b"\x00\x00\x01", index + 3)
    return False

class Count(Node):
    def __init__(self):
        super().__init__("count_h264_idr")
        self.idr = 0
        self.p = 0
        self.gaps = []
        self.since_idr = None
        self.create_subscription(CompressedImage, TOPIC, self.on_image, 10)

    def on_image(self, message):
        if "h264" not in message.format.lower() and "avc" not in message.format.lower():
            print(f"不是 H264: format={message.format!r}")
            return
        if is_idr(bytes(message.data)):
            if self.since_idr not in (None, 0):
                self.gaps.append(self.since_idr)
            self.idr += 1
            self.since_idr = 0
        elif self.since_idr is not None:
            self.p += 1
            self.since_idr += 1

rclpy.init()
node = Count()
end = time.monotonic() + SECONDS
while time.monotonic() < end and rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node()
rclpy.shutdown()

total = node.idr + node.p
print(f"topic {TOPIC}")
print(f"seconds {SECONDS:.0f}")
print(f"packets {total}")
print(f"idr {node.idr}")
print(f"p {node.p}")
if node.gaps:
    print(f"p_frames_between_idr {node.gaps}")
    print(f"p_between_idr_min {min(node.gaps)} max {max(node.gaps)}")
PY
```

`p_frames_between_idr` 是相邻两个 IDR 之间的 P 帧个数。例如输出 `29`，就是 1 个 IDR 后面跟 29 个 P 帧，大约每秒一个 IDR（相机若是 30 Hz）。第一个 IDR 之前的包不计入，因为客户端也会丢掉它们。

30 Hz 相机常见的默认是 1 个 IDR 加 29 个 P 帧。跑完上面的脚本之前，不要把这个默认当成这台机器人的结果。

## 和推理的关系

策略不读取这个比例。一次推理用的是队列里攒下的最新画面，不是“必须 5 个包”。

客户端在 `vla_client/openpi_client_gento.py` 里的规则是：

1. 还没见过 IDR 时，包先丢掉，服务端也解不出图。
2. IDR 到达后清空队列，从这个 IDR 重新追加。
3. 发起推理时，把队列里现有的包一次发给服务端，然后清空。

所以第一次请求里的包数，是“从上一个 IDR 到按下推理之间相机到了多少帧”，不是固定的 5。那次假数据用 5，只是在模仿启动阶段已经积压了 1 个 IDR 和 4 个 P 帧。
