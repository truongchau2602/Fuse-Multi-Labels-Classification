import pickle
import pandas as pd
import numpy as np

path = "/content/drive/MyDrive/DistributionBalancedLoss/OLIVES_result.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

binary_pred_list = []
for prediction in data[0]["outputs"]:
    prob_prediction = 1/(1+np.exp(-prediction))
    binary_pred_list.append(prob_prediction)


development_path = "/content/drive/MyDrive/IEEE_2023_Ophthalmic_Biomarker_Det/TEST/test_set_submission_template.csv"
development = pd.read_csv(development_path)

CLASSES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']


binary_pred_df = pd.DataFrame(binary_pred_list, columns=CLASSES)

path_names = development['Path (Trial/Image Type/Subject/Visit/Eye/Image Name)']
binary_pred_df = pd.concat([path_names, binary_pred_df], axis=1)

binary_pred_df.to_csv('submit_OLIVES_epoch_2.csv', index = False)
# ./work_dirs/LT_swinTrans_Transformer_DB_train_with_flag_final_1/latest.pth