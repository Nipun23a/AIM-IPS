import pandas as pd

df = pd.read_csv("./mapped_data/cicids_mapped.csv")

features = [
    "flow_duration","total_fwd_packets","total_bwd_packets",
    "total_fwd_bytes","total_bwd_bytes","flow_bytes_per_sec",
    "flow_pkts_per_sec","fwd_pkts_per_sec","bwd_pkts_per_sec",
    "pkt_len_mean","pkt_len_std","pkt_len_max","pkt_len_min",
    "flow_iat_mean","flow_iat_std","flow_iat_max",
    "fwd_iat_mean","bwd_iat_mean",
    "fin_flag_count","syn_flag_count","rst_flag_count",
    "psh_flag_count","ack_flag_count",
    "init_win_fwd","init_win_bwd",
    "fwd_ttl_mean","bwd_ttl_mean",
    "fwd_pkt_len_mean","bwd_pkt_len_mean",
    "fwd_pkt_len_std","bwd_pkt_len_std",
    "down_up_ratio"
]

print(df[features].describe())