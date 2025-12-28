# Probe.py
import sys
import os
from probeinterface.plotting import plot_probe

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import probeinterface as pi
def channel_128_location( ):
    import pandas as pd

# Load the Excel file
    file_path = '/home/asus/Desktop/Shujun_code/30channels_250707/data/128channels_tip_1_1126.xlsx'
    excel_data = pd.read_excel(file_path)

# Display the first few rows of the Excel data to understand its structure
    excel_data.head()
# Display the first few rows of the Excel data to understand its structure
    print(excel_data.head())

# Extract and sort the coordinates by channel id in both sets
# Filter out rows where 'channel id' or 'channel id.1' might be NaN
    locations = []

# Process the first set of coordinates
    # this channel id is headstage id!!!!
    first_set = excel_data[['Contact_ID', 'X_um', 'Y_um']].dropna()
    first_coords = first_set[['X_um', 'Y_um']].values.tolist()
    contact_id = first_set[['Contact_ID']].values.tolist()

# Combine both sets of coordinates
    locations = first_coords
    return(locations,contact_id)

def build_probe_128ch():
    """
    构建并返回一个配置好的 Probe 对象
    """
    # 读取电极位置
    locations,contact_ids = channel_128_location()
    #print("Locations:", locations)


    # 探针几何编号，用于 annotate 可视化
    import numpy as np
    contact_ids = np.array([str(i) for i in range(128)]) 


    # 创建 Probe 对象
    probe = pi.Probe(ndim=2)
    probe.set_contacts(
        positions=locations,
        shapes='circle',
        contact_ids=contact_ids,
        shape_params={'radius': 20}  # 单位：um
    )

    # 设置采集设备的物理通道映射
    probe.set_device_channel_indices(contact_ids)

    # 添加可视化标识
    probe.annotate(contact_ids=contact_ids)

    # 打印检查
    print("Probe:", probe)
    print("Probe positions:", probe.contact_ids)
    return probe





probe = build_probe_128ch()
from probeinterface.plotting import plot_probe
import matplotlib.pyplot as plt
plot_probe(probe)  # 可视化检查
plt.show()