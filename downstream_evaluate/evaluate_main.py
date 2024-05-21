import numpy as np
import statistics
from evaluate_air2cold import eva_air2cold
from evaluate_cold2air import eva_cold2air
from evaluate_air2air import eva_air2air
from evaluate_cold2cold import eva_cold2cold

num_dataset = 2 #7
# dataset =[2,3,4,7,8,9,10]
c2a_all_wrong = np.zeros(num_dataset)
all_edges = np.zeros(num_dataset)
a2c_all_wrong = np.zeros(num_dataset)
a2c_all_right = np.zeros(num_dataset)
a2a_all_wrong = np.zeros(num_dataset)
a2a_all_right = np.zeros(num_dataset)
c2c_all_wrong = np.zeros(num_dataset)
c2c_all_right = np.zeros(num_dataset)
c2a_all_wrong_percent = np.zeros(num_dataset)
a2c_all_right_percent = np.zeros(num_dataset)
a2a_all_right_percent = np.zeros(num_dataset)
c2c_all_right_percent = np.zeros(num_dataset)
node_path = "./downstream_evaluate/node_name.csv" # "node_name.csv"
a2c_label_path = "./downstream_evaluate/air2cold_label.csv" # "air2cold_label.csv"
a2a_label_path = "./downstream_evaluate/air2air_label.csv" # "air2cold_label.csv"
c2c_label_path = "./downstream_evaluate/cold2cold_label.csv" # "air2cold_label.csv"
    
for i in range(num_dataset):
    # data_path = f"./exp_real/downstream/dataset{i+1}/train/DAG.npy"
    # data_path = f"./exp_real/downstream_llm/dataset{i+1}/train/DAG.npy"
    # data_path = f"./exp_real/downstream_unknown_tecdixing/dataset{i+1}/train/DAG.npy"
    # data_path = f"./exp_real/downstream_tecd_obsinterv/dataset{i+1}/train/DAG.npy"
    # data_path = f"./exp_real/dw_obs_no_interv/dataset{i+1}/train/DAG.npy"
    # data_path = f"./baseline/DYNOTEARS/downstream/DAG{i+1}.npy"
    # data_path = f"./baseline/PCMCI/baseline_results/downstream/PCMCI_est{i+1}.npy"
    # data_path = f"./exp_real/downstream/dataset{i+1}/train/DAG.npy"
    # data_path = f"./baseline/NeuralGC/results/downstream_llm/NeuralGC_baseline/GC_est{i+1}.npy"
    # data_path = f"./baseline/NeuralGC/results/downstream_llm/dcdi_results_transfer/GC_est{i+1}.npy"

    data_path = './llm_res/gpt3.5turboinstructorbasep_res_matrix.npy'
    c2a_wrong, all_edge = eva_cold2air(data_path, node_path)
    c2a_all_wrong[i] = c2a_wrong
    c2a_all_wrong_percent[i] = c2a_wrong/all_edge
    all_edges[i] = all_edge
    a2c_wrong, a2c_right = eva_air2cold(data_path, node_path, a2c_label_path)
    # a2c_all_wrong[i] = a2c_wrong
    a2c_all_right[i] = a2c_right
    a2c_all_right_percent[i] = a2c_right/all_edge
    a2a_wrong, a2a_right = eva_air2air(data_path, node_path, a2a_label_path)
    # a2a_all_wrong[i] = a2a_wrong
    a2a_all_right[i] = a2a_right
    a2a_all_right_percent[i] = a2a_right/all_edge
    c2c_wrong, c2c_right = eva_cold2cold(data_path, node_path, c2c_label_path)
    # c2c_all_wrong[i] = c2c_wrong
    c2c_all_right[i] = c2c_right
    c2c_all_right_percent[i] = c2c_right/all_edge

print('all_edges:',round(np.mean(all_edges),2),'\pm',round(statistics.stdev(all_edges),2))
print('c2a_all_wrong:',round(np.mean(c2a_all_wrong),2),'\pm',round(statistics.stdev(c2a_all_wrong),2))
print('c2a_all_wrong_percent:',round(100*np.mean(c2a_all_wrong_percent),2),'\pm',round(100*statistics.stdev(c2a_all_wrong_percent),2))
# print('a2c_all_wrong:',round(np.mean(a2c_all_wrong),2),'\pm',round(statistics.stdev(a2c_all_wrong),2))
print('a2c_all_right:',round(np.mean(a2c_all_right),2),'\pm',round(statistics.stdev(a2c_all_right),2))
print('a2c_all_right_percent:',round(100*np.mean(a2c_all_right_percent),2),'\pm',round(100*statistics.stdev(a2c_all_right_percent),2))
# print('a2a_all_wrong:',round(np.mean(a2a_all_wrong),2),'\pm',round(statistics.stdev(a2a_all_wrong),2))
print('a2a_all_right:',round(np.mean(a2a_all_right),2),'\pm',round(statistics.stdev(a2a_all_right),2))
print('a2a_all_right_percent:',round(100*np.mean(a2a_all_right_percent),2),'\pm',round(100*statistics.stdev(a2a_all_right_percent),2))
# print('c2c_all_wrong:',round(np.mean(c2c_all_wrong),2),'\pm',round(statistics.stdev(c2c_all_wrong),2))
print('c2c_all_right:',round(np.mean(c2c_all_right),2),'\pm',round(statistics.stdev(c2c_all_right),2))
print('c2c_all_right_percent:',round(100*np.mean(c2c_all_right_percent),2),'\pm',round(100*statistics.stdev(c2c_all_right_percent),2))