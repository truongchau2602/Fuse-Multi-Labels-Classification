import pickle
import pandas as pd
import numpy as np


path ="/content/drive/MyDrive/OOD_CV_Track1/DistributionBalancedLoss/ood_result.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

binary_pred_list = []
for prediction in data[0]["outputs"]:
    prob_prediction = 1/(1+np.exp(-prediction))
    binary_pred_list.append(prob_prediction)

development_path = "/content/drive/MyDrive/OOD_CV_Track1/train/test.csv"
development = pd.read_csv(development_path)

pathology = pd.read_csv("/content/drive/MyDrive/OOD_CV_Track1/train/labels.csv")
# pathology = pathology[9:-1]
# print(pathology)
CLASSES = ['aeroplane', 'bicycle', 'boat', 'bus',
            'car', 'chair', 'diningtable',
            'motorbike', 'sofa', 'train']

# binary_prediction_list = np.array(binary_pred_list)

# for binary_prediction in binary_pred_list:
#     print(binary_prediction)

# assert(binary_prediction_list.shape[0] == len(development))

binary_pred_df = pd.DataFrame(binary_pred_list, columns=CLASSES)

imgs = development['imgs']
binary_pred_df = pd.concat([imgs, binary_pred_df], axis=1)

save_path = "/content/drive/MyDrive/OOD_CV_Track1/DistributionBalancedLoss/csv_submission/submit_ood.csv"
binary_pred_df.to_csv(save_path, index = False)
# ./work_dirs/LT_swinTrans_Transformer_DB_train_with_flag_final_1/latest.pth