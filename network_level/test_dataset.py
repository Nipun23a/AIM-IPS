import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("./mapped_data/cicids_mapped.csv")

model = joblib.load("./models/lightgbm_network.pkl")
le = joblib.load("./models/label_encoder.pkl")

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

sample = df.sample(20)

X = sample[features]

probs = model.predict(X)

pred_idx = np.argmax(probs, axis=1)

pred_labels = le.inverse_transform(pred_idx)

print("Actual:", sample["unified_label"].values)
print("Predicted:", pred_labels)