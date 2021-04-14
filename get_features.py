import h5py
import dpkt
import pandas as pd

file_path ="/home/svetlana/Documents/git/packet2vec/output/features/t9_features.h5"

data = h5py.File(file_path, 'r')['vectors']

def get_label(pack):
    # Find incident description where time matches
    idx_time = alerts.Interval.apply(lambda iv: row_df.timestamp in iv)

    if not idx_time.any():
        return 0
    
    # Find  incident description where at least one IP matches:
    ignored_ips = {
        '205.174.165.80', # External IP of firewall (NATing)
        '172.16.0.1', # Internal IP of firewall (NATing)
    }
    idx_ip = pd.Series(np.zeros(alerts.shape[0], dtype=bool), index=alerts.index)
    
    if row_df["ip.src"] not in ignored_ips:
        idx_ip = idx_ip | (row_df["ip.src"] == alerts['IP attacker']) | (row_df["ip.src"] == alerts['IP victim'])
    if row_df["ip.dst"] not in ignored_ips:
        idx_ip = idx_ip | (row_df["ip.dst"] == alerts['IP attacker']) | (row_df["ip.dst"] == alerts['IP victim'])
    
    idx = idx_time & idx_ip
    # No hits then default to benign
    if not idx.any():
        return 0
    else:
        return 1

labels = 

    
