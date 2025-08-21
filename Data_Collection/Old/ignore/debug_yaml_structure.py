import yaml
import pandas as pd

# Let's examine the actual structure of the YAML data
with open('4651e371-2b19-42ba-af9c-90ea170ce564.xes.yaml', 'r') as f:
    docs = list(yaml.safe_load_all(f))

print("Total documents:", len(docs))

# Examine first few events to understand structure
for i, doc in enumerate(docs[:10]):
    print(f"\n--- Document {i} ---")
    if 'event' in doc:
        event = doc['event']
        print(f"Event keys: {list(event.keys())}")
        
        if 'data' in event and event['data']:
            print("Data items:")
            for j, data_item in enumerate(event['data']):
                if data_item:
                    print(f"  {j}: {data_item}")
        
        # Look for activity names that might indicate environmental data collection
        if 'concept:name' in event:
            print(f"Activity: {event['concept:name']}")
        if 'concept:endpoint' in event:
            print(f"Endpoint: {event['concept:endpoint']}")
            
    print("-" * 40)
