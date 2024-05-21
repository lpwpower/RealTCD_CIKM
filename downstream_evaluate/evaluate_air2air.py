import numpy as np
import pandas as pd

def eva_air2air(data_path, node_path, label_path):
    data = np.load(data_path)
    # print("data shape:", data.shape)
    
    node_name = pd.read_csv(node_path, index_col=0)["0"].tolist()
    # print("node num:", len(node_name))

    label = pd.read_csv(label_path, index_col=0)
    # print("node num:", len(node_name))
    
    data_extract = data[:, :38]
    # print(data_extract.shape)

    if data_extract.shape[0]==38:
        df_data_extract = pd.DataFrame(data_extract, index=node_name, columns=node_name)
    else:
        df_data_extract = pd.DataFrame(data_extract, index=node_name+node_name, columns=node_name)
    # print(df_data_extract.head())

    wrong_num = 0
    right_num = 0

    for i in range(len(df_data_extract)):
        row = df_data_extract.iloc[i]
        name = row.name  
        
        if "送风温度" in name:  
            air2air_label = label.loc[name]   
            
            selected_label = air2air_label[air2air_label==1].index.tolist()  

            label_counts = row[selected_label]   
            
            right_num += len(label_counts[label_counts==1])  
            wrong_num += len(label_counts[label_counts==0]) 

    print("wrong_num:", wrong_num, "right_num:", right_num)

    
    return wrong_num, right_num


if __name__ == "__main__":
    all_wrong_nums = 0
    all_right_nums = 0

    node_path = "data_real/downstream/node_name.csv" # "node_name.csv"
    label_path = "/home/lipeiwen.lpw/TECDI/downstream_evaluate/air2cold_label.csv" # "air2cold_label.csv"
    
    for i in range(1,11):
    # for i in [6]:
        # data_path = f"./downstream_less2/GC_est{i}.npy"
        # data_path = f"./downstream_less/dataset{i}/train/DAG.npy"
        # data_path = f"./dynotears_downstream_less/DAG{i}.npy"
        # data_path = f"./PCMCI_downstream_less/result_data{i}.npy"
        data_path = f'/home/lipeiwen.lpw/TECDI/baseline/DYNOTEARS/downstream/DAG{i}.npy'
        # data_path = f'/home/lipeiwen.lpw/TECDI/exp_real/downstream_old/dataset{i}/train/DAG.npy'

        wrong_num, right_num = eva_air2air(data_path, node_path, label_path)
        
        all_wrong_nums += wrong_num
        all_right_nums += right_num

    print('all_wrong_nums:', all_wrong_nums, "all_right_num:", all_right_nums)
    


