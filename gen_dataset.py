import pandas as pd
import numpy as np
import dpkt
import pdb
import datetime
import collections
import socket
import dpkt 

path_file = 'Data/cic_ids_2017/PCAPs/Thursday-WorkingHours.pcap'

f = open(path_file, 'rb')
packets = dpkt.pcapng.Reader(f)

nbr_print = 10000
def print_avancement(i, ts=None, pcks=None, zone=0):
    if i % nbr_print == 0:
        print("i: " + str(i))
        if ts != None:
            print(str(delta_format(ts)))
        if pcks != None:
            print("nbr 0 " + str(len(pcks[False])))
            print("nbr 1 " + str(len(pcks[True])))
        print("zone " + str(zone) + "\n\n")

def delta_format(ts):
    dt_ts = datetime.datetime.utcfromtimestamp(ts)
    return  dt_ts - datetime.timedelta(hours=3) # in my case it's 3h because my first packet has as datetime 11:58:58 and the begin of capture is ~9am

def recolt_feat(feat, path, obj, i):
    root = path.split(".")[0]
    if root in dir(obj):
        value = obj.__getattribute__(root)
        if not("." in path): # ca veut dire qu'on a trouvé
            return True, value
        
        return recolt_feat(feat, ".".join(path.split(".")[1:]), value, i)
    
    return False, None

# importing data
alerts = pd.read_csv('Data/cic_ids_2017/incident_descriptions.csv', 
    parse_dates= ['Start', 'Stop'],
    date_parser = lambda x: datetime.datetime.strptime(x, '%d-%m-%y %H:%M'),
    sep='\t'
)

alerts['Stop'] = alerts['Stop'].dt.floor('min') + datetime.timedelta(minutes=1)
alerts['Interval'] = alerts[['Start', 'Stop',]].apply(
    lambda d: pd.Interval(d[0], d[1], closed='both'), 
    axis=1
)

print("get incidents ok")

def get_label(row_df):    
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
    
    # No hits then default to benign
    if not idx_ip.any():
        return 0
    else:
        return 1


pcks = {True: [], False: []} 


def in_attack_zone(tps, index_start):
    zones = [("06-07-17 09:20", "06-07-17 10:00"),
    ("06-07-17 10:15", "06-07-17 10:35"),
    ("06-07-17 10:40", "06-07-17 10:42"),
    ("06-07-17 14:19", "06-07-17 14:19"),
    ("06-07-17 14:20", "06-07-17 14:21"),
    ("06-07-17 14:33", "06-07-17 14:35"),
    ("06-07-17 14:53", "06-07-17 15:00"),
    ("06-07-17 15:04", "06-07-17 15:45")]

    for i, (start_, end_) in enumerate(zones[index_start:]):
        start = datetime.datetime.strptime(start_, '%d-%m-%y %H:%M')
        end = datetime.datetime.strptime(end_, '%d-%m-%y %H:%M')

        if start <= tps <= end:
            return True, index_start + i

        if start > tps:
            break

    return False, index_start

index_start_attack_zone = 0
for i, (ts, buf) in enumerate(packets):  
    
    print_avancement(i, ts, pcks, index_start_attack_zone)

    is_in_attack_zone, new_index_attack_zone = in_attack_zone(delta_format(ts), index_start_attack_zone)
    index_start_attack_zone = new_index_attack_zone
    if not is_in_attack_zone:
        pcks[False].append((i, ts, buf))
        continue
    
    packet_to_add = {}  
    pck = dpkt.ethernet.Ethernet(buf)
    path = "ip.src"
    get, value = recolt_feat(path, path, pck, i)
    if not get:
        pcks[False].append((i, ts, buf))
        continue
    packet_to_add[path] = socket.inet_ntop(socket.AF_INET, value)
    path = "ip.dst"
    get, value = recolt_feat(path, path, pck, i)
    if not get:
        pcks[False].append((i, ts, buf))
        continue
    packet_to_add[path] = socket.inet_ntop(socket.AF_INET, value)

    label = get_label(packet_to_add)
    
    pcks[label].append((i, ts, buf))


norms_file =  dpkt.pcap.Writer(open("Data/cic_ids_2017/norms_all.pcap" ,'wb'))
labels = {"Label": []}
for i, ts, buf in pcks[False]:
    labels["Label"].append(0)
    norms_file.writepkt(buf, ts)

outs_file =  dpkt.pcap.Writer(open("Data/cic_ids_2017/outs_all.pcap" ,'wb'))
for i, ts, buf in pcks[True]:
    labels["Label"].append(1)
    outs_file.writepkt(buf, ts)


data = pd.DataFrame.from_dict(labels)
data.to_csv("Data/cic_ids_2017/labels_all.csv", sep=';')
