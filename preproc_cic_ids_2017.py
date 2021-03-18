import pandas as pd
import numpy as np
import dpkt
import pdb
import datetime
import pickle
import collections
import socket

path_file = 'Data/cic_ids_2017/sub_pcap/sub_Thurday_00000_20170706135858.pcap'
#path_file = 'Data/cic_ids_2017/sub_pcap/Thursday-WorkingHours.pcap'
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
def print_avancement(i, ts):
    if i % nbr_print == 0:
        print("i: " + str(i))
        print(str(delta_format(ts)))

def delta_format(ts):
    dt_ts = datetime.datetime.utcfromtimestamp(ts)
    to_print = True
    if to_print:
        print(dt_ts)
    return  dt_ts - datetime.timedelta(hours=3) # in my case it's 3h because my first packet has as datetime 11:58:58 and the begin of capture is ~9am


def go_feat(b, i, ts):
    pck = dpkt.ethernet.Ethernet(b)
    recolt_features(pck, i)
    set_feature("timestamp", i, ts, format_method=delta_format)
    print_avancement(i, ts)

def go_features():

    threads = set()
    for i, (ts, buf) in enumerate(packets):    
        if i == 1:
            break    

        go_feat(buf, i, ts)

go_features()
df = pandas.DataFrame(hierarchie, columns=hierarchie.keys())

# importing data
fields = ['Start', 'Stop']
alerts = pd.read_csv('Data/cic_ids_2017/incident_descriptions.csv', 
    parse_dates=fields,
    usecols=fields,
    date_parser = lambda x: datetime.datetime.strptime(x, '%d-%m-%y %H:%M')
)

class AttackError(Exception):
    pass

# for the moment, we concentrate on only web attack
ip_attackers = ["205.174.165.73"]
ips_victims = ["205.174.165.68", "192.168.10.50"]

def get_attack(ts, start, end, ip_src, ip_dst):
    if start <= ts <= end and ((ip_src in ip_attacker and ip_dst in ips_victims) or (ip_src in ips_victims and ip_dst in ip_attacker)):
        raise AttackError

def recolt_label(x):
    try:
        ip_src = socket.inet_ntop(socket.AF_INET, x['ip.src'])
        ip_dst = socket.inet_ntop(socket.AF_INET, x['ip.dst'])
        ts = x['timestamp']
        try:
            alerts.apply(lambda x: get_attack(ts, x['Start'], x['Stop'], ip_src, ip_dst), axis=1)
            return 0
        except AttackError:
            return 1 # it's an attack !
    except KeyError:
        return 0

df['label'] = df.apply(recolt_label, axis=1)

df.to_csv("pcap.csv")