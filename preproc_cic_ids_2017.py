import pandas as pd
import numpy as np
import dpkt
import pdb
import datetime
import collections
import socket

path_file = 'Data/cic_ids_2017/Thursday-WorkingHours.pcap'

f = open(path_file, 'rb')
packets = dpkt.pcapng.Reader(f)

hierarchie = {}
features_ids = {}

def set_feature(feature, i, value, format_method=lambda x:x):
    if feature in hierarchie:
        features_ids[feature].add(i) 
        hierarchie[feature][i] = format_method(value)
    else:
        hierarchie[feature] = collections.OrderedDict()
        features_ids[feature] = set()
        set_feature(feature, i, value, format_method)


def recolt_features(obj, i, path="p"):
    for attr_str in dir(obj):
        if attr_str.startswith("_") or "data" in attr_str:
            continue
        try:
            attr_value = obj.__getattribute__(attr_str)
        except:
            continue
        type_attr = str(type(attr_value))
        if "method" in type_attr:
            continue
        new_path = path + "." + attr_str
        if "dpkt" in type_attr:
            recolt_features(attr_value, i, new_path)
        else:
            new_path = new_path[2:] # remove "p."
            set_feature(new_path, i, attr_value)

nbr_print = 10000
def print_avancement(i, ts=None):
    if i % nbr_print == 0:
        print("i: " + str(i))
        if ts != None:
            print(str(delta_format(ts)))

def delta_format(ts):
    dt_ts = datetime.datetime.utcfromtimestamp(ts)
    to_print = False
    if to_print:
        print(dt_ts)
    return  dt_ts - datetime.timedelta(hours=3) # in my case it's 3h because my first packet has as datetime 11:58:58 and the begin of capture is ~9am

def recolt_feat(feat, path, obj, i):
    root = path.split(".")[0]
    if root in dir(obj):
        value = obj.__getattribute__(root)
        if not("." in path): # ca veut dire qu'on a trouvé
            if feat in hierarchie.keys():
                hierarchie[feat][i] = value
            else:
                hierarchie[feat] = collections.OrderedDict()
                hierarchie[feat][i] = value
        else: # On doit rentrer plus en profondeur
            recolt_feat(feat, ".".join(path.split(".")[1:]), value, i)
    else:
        if feat in hierarchie.keys():
            hierarchie[feat][i] = np.NaN
        else:
            hierarchie[feat] = collections.OrderedDict()
            hierarchie[feat][i] = np.NaN

def go_feat(b, i, ts):
    pck = dpkt.ethernet.Ethernet(b)
    #recolt_features(pck, i)
    path = "ip.src"
    recolt_feat(path, path, pck, i)
    path = "ip.dst"
    recolt_feat(path, path, pck, i)
    set_feature("timestamp", i, ts, format_method=delta_format)
    print_avancement(i, ts)

def go_features():
    for i, (ts, buf) in enumerate(packets):    
        go_feat(buf, i, ts)
        
go_features()
df = pd.DataFrame(hierarchie, columns=hierarchie.keys())

# importing data
alerts = pd.read_csv('Data/cic_ids_2017/incident_descriptions.csv', 
    parse_dates=['Start', 'Stop'],
    date_parser = lambda x: datetime.datetime.strptime(x, '%d-%m-%y %H:%M')
)

alerts['Stop'] = alerts['Stop'].dt.floor('min') + datetime.timedelta(minutes=1)
alerts['Interval'] = alerts[['Start', 'Stop',]].apply(
    lambda d: pd.Interval(d[0], d[1], closed='both'), 
    axis=1
)

def get_label(row_df):
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

print("ip src in progress")
df['ip.src'] = df['ip.src'].apply(lambda x: socket.inet_ntop(socket.AF_INET, x) if x == x else x) # the if statement is here for the case where x = nan
print("ip dst in progress")
df['ip.dst'] = df['ip.dst'].apply(lambda x: socket.inet_ntop(socket.AF_INET, x) if x == x else x) 


df['Label'] = df.apply(get_label, axis=1)

timestamps = df.pop("timestamp")
df.insert(0, "timestamp", timestamps)

df.to_csv("Data/cic_ids_2017/pcap_light.csv")
