from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd

# 1. 指定事件文件路径
event_file = "F:\thesis_code\log_CL\99_98_97\seq\epochs=150_seed=44\events.out.tfevents.1744704090.pc3054.3669540.0"

# 2. 加载：告诉它 scalars 全都读出来（0 表示不限制数量）
ea = event_accumulator.EventAccumulator(
    event_file,
    size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()

# 3. 查看有哪些 scalar tags
print("可用 tags:", ea.Tags()["scalars"])

# 4. 取出 train_loss 对应的所有条目
#    每个 entry 是一个 namedtuple： (wall_time, step, value)
entries = ea.Scalars("train_loss")

# 5. 转成列表 / NumPy /DataFrame
steps      = [e.step for e in entries]
wall_times = [e.wall_time for e in entries]
values     = [e.value for e in entries]

# 作为 Python 列表
loss_list = values

# 或转 NumPy 数组
loss_arr = np.array(values)

# 或转成 Pandas DataFrame
df = pd.DataFrame({
    "step": steps,
    "wall_time": wall_times,
    "train_loss": values
})

print(df.head())
