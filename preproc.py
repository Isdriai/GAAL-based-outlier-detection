# Pour reprendre le format des données qu'attend GAAL, il faut ajouter un id, traiter les données catégoriques 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def to_one_hot(features, data):
    data_copy = data.copy()
    for f in features:
        dum = pd.get_dummies(data_copy[[f]])
        data_copy = pd.concat([data_copy, dum], axis=1).drop(f, axis=1)
    return data_copy

def move_column(name_col, new_pos, data):
    col = data.pop(name_col)
    data.insert(new_pos, name_col, col)

columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
"count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class"]

path = "Data/nsl-kdd/"
data = pd.read_table(path + "KDD", sep=',', names=columns, dtype="unicode")
data = data.iloc[1:,:]

data = to_one_hot(["service", "protocol_type", "flag"], data)

for i in range(len(data)):
    if data["class"].values[i] == "normal": 
        data["class"].values[i] = "nor"
    else:
        data["class"].values[i] = "out" 

data["id"] = data.index
move_column("id", 0, data)
move_column("class", 1, data)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.iloc[:,2:])
data = np.concatenate([data.iloc[:,:2], data_scaled], axis=1)

pd.DataFrame(data).to_csv(path + "KDDproc", header=False, index=False)

