import os
import glob
import pandas as pd
import numpy as np
import h5py

# 指定 CSV 文件所在的目录
csv_folder = r"D:\Projects\openfoamExportData\case1_validation\process_LR_Geo"
# 获取所有 CSV 文件的路径
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

# 定义保存 HDF5 文件的路径
output_dir = "../../dataset"
os.makedirs(output_dir, exist_ok=True)
hdf5_file = os.path.join(output_dir, "case1LR_generalization_Geo.h5")

# 创建 HDF5 文件
with h5py.File(hdf5_file, 'w') as f:
    for i, file in enumerate(csv_files):
        # 读取 CSV 文件
        df = pd.read_csv(file)
        # 将 DataFrame 转换为 numpy 数组
        data = df.to_numpy()
        # 创建数据集，名称为 'sample_i'，其中 i 为样本索引
        f.create_dataset(f'sample_{i}', data=data, compression="gzip")
