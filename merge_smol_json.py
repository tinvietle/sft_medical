import os
from pathlib import Path
import json

def merge_jsons_files(input_folder, output_file):
    input_path = Path(input_folder)
    all_data = []
    
    for json_file in input_path.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            all_data.append(data)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
input_folder = "smol_json"
output_file = "smol_merged.json"
merge_jsons_files(input_folder, output_file)
            
            
            