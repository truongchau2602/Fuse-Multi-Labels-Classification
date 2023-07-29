import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file csv vào DataFrame
df = pd.read_csv('/content/drive/MyDrive/IEEE_2023_Ophthalmic_Biomarker_Det/TRAIN/Training_Biomarker_Data.csv')
print(df.columns)
# Tính số lượng hàng có giá trị 1 trong từng cột
counts = df.iloc[:, 1:].sum().sort_values()

# Lấy tên các cột đã sắp xếp
sorted_columns = counts.index.tolist()

# Vẽ biểu đồ cột theo thứ tự tăng dần
ax = counts.plot(kind='bar', rot=0)

# Cài đặt tiêu đề và nhãn trục
ax.set_title('Phân phối giá trị 1 trong các cột')
ax.set_xlabel('Tên cột')
ax.set_ylabel('Số lượng giá trị 1')

# In số lượng giá trị 1 trên mỗi cột
for i, count in enumerate(counts):
    ax.text(i-0.2, count+0.1, str(count), ha='center', rotation='vertical')

# Cho tên cột nằm nghiêng
ax.set_xticklabels(sorted_columns, rotation="vertical", ha='right')

# Hiển thị biểu đồ
plt.show()