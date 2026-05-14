"""
Создание одного словаря культурных маркеров
"""

import csv
import re
import pandas as pd
import urllib.request
from difflib import SequenceMatcher

def clean_marker(text):
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    if len(text) < 2:
        return ""
    
    return text

def extract_marker_from_row(row_value, col_name):
    value = str(row_value).strip()
    
    if '>>' in value:
        marker_part = value.split('>>')[0].strip()
        marker_part = marker_part.strip('{}').strip('[]').strip('()')
        return marker_part
    
    return value

def is_similar(a, b, threshold=0.85):
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold

def remove_similar_duplicates(markers, threshold=0.85):
    if not markers:
        return []
    
    result = []
    skip_indices = set()
    
    for i, marker1 in enumerate(markers):
        if i in skip_indices:
            continue
        
        best = marker1
        
        for j, marker2 in enumerate(markers[i+1:], i+1):
            if is_similar(marker1, marker2, threshold):
                skip_indices.add(j)
                if len(marker2) < len(best):
                    best = marker2
        
        result.append(best)
    
    return result

def load_markers_from_your_csv(filepath):
    markers = []
    
    try:
        df = pd.read_csv(filepath)
        
        first_col = df.columns[0]
        
        for val in df[first_col].dropna():
            marker = extract_marker_from_row(val, first_col)
            cleaned = clean_marker(marker)
            if cleaned:
                markers.append(cleaned)
        
        print(f"Загружено {len(markers)} маркеров из {filepath}")
        return markers
        
    except Exception as e:
        print(f"Ошибка загрузки {filepath}: {e}")
        return []

def download_idioms():
    idioms = []
    
    url = "https://raw.githubusercontent.com/baiango/english_idioms/main/idioms.csv"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        
        lines = data.strip().split('\n')
        for line in lines[1:]:
            match = re.search(r'\{(.+?)\}', line)
            if match:
                idiom = match.group(1).strip()
                cleaned = clean_marker(idiom)
                if cleaned:
                    idioms.append(cleaned)
        
        print(f"Загружено {len(idioms)} идиом")
        return idioms
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return []

def merge_all_datasets():
    all_markers = []
    
    your_files = ["data/cultural_markers_dataset.csv", "data/cultural_markers_dataset_2.csv"]
    
    for file in your_files:
        try:
            markers = load_markers_from_your_csv(file)
            all_markers.extend(markers)
        except FileNotFoundError:
            print(f"  Файл {file} не найден")
    
    idioms = download_idioms()
    all_markers.extend(idioms)

    all_markers = [m for m in all_markers if m]
    
    unique_markers = list(set(all_markers))
    unique_markers.sort()
    
    final_markers = remove_similar_duplicates(unique_markers, threshold=0.85)
    
    filtered = []
    for marker in final_markers:
        if len(marker) < 2:
            continue
        if marker.isdigit():
            continue
        filtered.append(marker)
    
    filtered.sort()
    
    output_file = "data/cultural_markers_merged.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["cultural_marker"])
        for marker in filtered:
            writer.writerow([marker])
    
    print(f"Всего маркеров: {len(filtered)}")
    
    print("\n Первые 20 маркеров:")
    for i, marker in enumerate(filtered[:20], 1):
        print(f"{i}. {marker}")
    
    return filtered

if __name__ == "__main__":
    result = merge_all_datasets()