import pickle
import pandas as pd
import numpy as np

# path = "/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/DistributionBalancedLoss/latest.pkl"
# path = "/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/DistributionBalancedLoss/submit_sampling5.pkl"
path ="/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/DistributionBalancedLoss/submit_final_1_epoch_8.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

binary_pred_list = []
for prediction in data[0]["outputs"]:
    prob_prediction = 1/(1+np.exp(-prediction))
    binary_pred_list.append(prob_prediction)

# development_path = '/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/Data/label/development_with_size.csv'
development_path = "/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/Data/label/development.csv"
development = pd.read_csv(development_path)

pathology = pd.read_csv('/media/aivn2023/86c50d28-d521-419b-a569-3aab9993961f/media/ai2023/HungAn/Chau/Data/label/no_path.csv').columns
pathology = pathology[9:-1]
print(pathology)

# binary_prediction_list = np.array(binary_pred_list)

# for binary_prediction in binary_pred_list:
#     print(binary_prediction)

# assert(binary_prediction_list.shape[0] == len(development))

binary_pred_df = pd.DataFrame(binary_pred_list, columns=pathology)

dicom_id = development['dicom_id']
binary_pred_df = pd.concat([dicom_id, binary_pred_df], axis=1)

binary_pred_df.to_csv('submit_final_1_epoch_8.csv', index = False)
# ./work_dirs/LT_swinTrans_Transformer_DB_train_with_flag_final_1/latest.pth