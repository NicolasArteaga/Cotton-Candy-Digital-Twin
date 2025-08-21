#!/usr/bin/env python3
"""
Script to clean batch process files and create filtered YAML files.
"""

import os
import yaml
import glob
from pathlib import Path

def parse_index_file(index_path):
    """Parse index.txt to find Cottonbot processes."""
    cottonbot_processes = []
    
    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Cottonbot - Run with Data Collection' in line:
                # Extract UUID from line like: "Cottonbot - Run with Data Collection (c3fb0896-65cb-4956-807c-c3ce7d297171) - 61272"
                start = line.find('(') + 1
                end = line.find(')')
                if start > 0 and end > start:
                    uuid = line[start:end]
                    cottonbot_processes.append(uuid)
    
    return cottonbot_processes

def get_batch_and_stick_from_file(yaml_path):
    """Extract batch_number and stick from any event in the YAML file that contains both."""
    try:
        with open(yaml_path, 'r') as f:
            # Load all documents from the YAML file
            docs = list(yaml.safe_load_all(f))
            
            # Search through all events for one that contains both batch_number and stick
            for i, doc in enumerate(docs[1:], 1):  # Skip the log header
                if 'event' in doc and 'data' in doc['event']:
                    data = doc['event']['data']
                    batch_number = None
                    stick = None
                    
                    for item in data:
                        if item['name'] == 'batch_number':
                            batch_number = item['value']
                        elif item['name'] == 'stick':
                            stick = item['value']
                    
                    # If we found both, return them along with the event index
                    if batch_number is not None and stick is not None:
                        return batch_number, stick, i
                        
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
    
    return None, None, None

def should_keep_event(event_doc):
    """Determine if an event should be kept in the cleaned file."""
    if 'event' not in event_doc:
        return False
    
    event = event_doc['event']
    
    # Keep events with stream/data lifecycle transition
    if event.get('cpee:lifecycle:transition') == 'stream/data':
        return True
    
    # Keep specific dataelements/change events
    if event.get('cpee:lifecycle:transition') == 'dataelements/change':
        concept_name = event.get('concept:name', '')
        
        # Keep weight measurement events
        if 'Weigh the whole Cotton Candy' in concept_name:
            return True
        
        # Keep pressure measurement events
        if 'Measure the pressure of the three sides of the cotton candy' in concept_name:
            return True
        
        # Keep size measurement events - these are actually stream/data events
        # but we'll check for them here too just in case
        if 'Measure the size of the Cotton Candy' in concept_name:
            return True
    
    return False

def clean_process_file(input_path, output_dir, batch_number, stick, metadata_event_index):
    """Clean a process file and save as {batch_number}-{stick}-process.yaml"""
    
    output_filename = f"{batch_number}-{stick}-process.yaml"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(input_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        
        # Keep the first document (log header) and the metadata event
        cleaned_docs = [docs[0]]  # Keep the log header
        
        # Always keep the metadata event (the one with batch_number and stick)
        if metadata_event_index < len(docs):
            cleaned_docs.append(docs[metadata_event_index])
        
        for i, doc in enumerate(docs[1:], 1):  # Skip the first document (log header)
            # Skip the metadata event since we already added it
            if i == metadata_event_index:
                continue
            if should_keep_event(doc):
                cleaned_docs.append(doc)
        
        # Write the cleaned YAML
        with open(output_path, 'w') as f:
            for i, doc in enumerate(cleaned_docs):
                if i > 0:
                    f.write('---\n')
                yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
                f.write('\n')
        
        print(f"Created cleaned file: {output_filename} ({len(cleaned_docs)-1} events)")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_batch_directory(batch_dir):
    """Process all files in a batch directory."""
    index_path = os.path.join(batch_dir, 'index.txt')
    
    if not os.path.exists(index_path):
        print(f"No index.txt found in {batch_dir}")
        return
    
    print(f"\nProcessing {batch_dir}...")
    
    # Get list of Cottonbot processes from index
    cottonbot_processes = parse_index_file(index_path)
    print(f"Found {len(cottonbot_processes)} Cottonbot processes")
    
    for uuid in cottonbot_processes:
        yaml_path = os.path.join(batch_dir, f"{uuid}.xes.yaml")
        
        if os.path.exists(yaml_path):
            # Get batch number and stick from the file
            batch_number, stick, metadata_event_index = get_batch_and_stick_from_file(yaml_path)
            
            if batch_number is not None and stick is not None and metadata_event_index is not None:
                print(f"Processing {uuid}: batch={batch_number}, stick={stick} (metadata at event {metadata_event_index})")
                clean_process_file(yaml_path, batch_dir, batch_number, stick, metadata_event_index)
            else:
                print(f"Could not extract batch_number/stick from {uuid}")
        else:
            print(f"File not found: {yaml_path}")

def main():
    """Main function to process a specific batch directory."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python batch_cleaner.py <batch_number>")
        print("Example: python batch_cleaner.py 5")
        sys.exit(1)
    
    try:
        batch_number = int(sys.argv[1])
    except ValueError:
        print("Error: Batch number must be an integer")
        sys.exit(1)
    
    base_dir = "/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/Batches"
    batch_dir = os.path.join(base_dir, f"batch-{batch_number}")
    
    if os.path.exists(batch_dir):
        print(f"Processing batch-{batch_number}...")
        process_batch_directory(batch_dir)
        print(f"\nBatch-{batch_number} processing complete!")
    else:
        print(f"Error: batch-{batch_number} directory not found!")
        print(f"Available batches:")
        for item in sorted(os.listdir(base_dir)):
            if item.startswith('batch-') and os.path.isdir(os.path.join(base_dir, item)):
                print(f"  {item}")
        sys.exit(1)

if __name__ == "__main__":
    main()
