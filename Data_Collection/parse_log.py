import yaml
import pandas as pd

def extract_env_data(event):
    if 'data' in event:
        for d in event['data']:
            if d['name'] == 'env':
                return d['value']
    return None

def extract_weight(event):
    if 'data' in event:
        for d in event['data']:
            if d['name'] == 'weight':
                return float(d['value'])
    return None

with open('4651e371-2b19-42ba-af9c-90ea170ce564.xes.yaml', 'r') as f:
    docs = list(yaml.safe_load_all(f))

env_entries = []
weights = []
turn_on_ts = None
turn_off_ts = None
final_ts = None
start_env = None
last_env = None
cc_started = None
cc_stopped = None

for doc in docs:
    event = doc.get("event", {})
    timestamp = event.get("time:timestamp", "")
    cpee_lifecycle = event.get("cpee:lifecycle:transition", "")
    activity = event.get("cpee:activity", "")

    env = extract_env_data(event)
    if env:
        env["timestamp"] = timestamp
        env_entries.append(env)
        if not start_env:
            start_env = env
        last_env = env

    if "turn_on" in str(event):
        turn_on_ts = timestamp
    if "turn_off" in str(event):
        turn_off_ts = timestamp

    weight = extract_weight(event)
    if weight:
        weights.append((timestamp, weight))
        if weight > 3 and not cc_started:
            cc_started = env
        elif weight < 2 and cc_started and not cc_stopped:
            cc_stopped = env

# Heuristic fallback if missing
after_on = next((e for e in env_entries if turn_on_ts and e["timestamp"] >= turn_on_ts), None)
before_end = last_env

vector = {
    "start_EnvH": start_env.get("EnvH") if start_env else None,
    "start_EnvT": start_env.get("EnvT") if start_env else None,
    "afterOn_EnvH": after_on.get("EnvH") if after_on else None,
    "afterOn_EnvT": after_on.get("EnvT") if after_on else None,
    "ccStart_EnvH": cc_started.get("EnvH") if cc_started else None,
    "ccStart_EnvT": cc_started.get("EnvT") if cc_started else None,
    "ccStop_EnvH": cc_stopped.get("EnvH") if cc_stopped else None,
    "ccStop_EnvT": cc_stopped.get("EnvT") if cc_stopped else None,
    "end_EnvH": before_end.get("EnvH") if before_end else None,
    "end_EnvT": before_end.get("EnvT") if before_end else None,
}

df = pd.DataFrame([vector])
df.to_csv("features.csv", index=False)
print("Feature vector saved to features.csv")