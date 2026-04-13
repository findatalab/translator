# Выявление аллюзий (модель: mistral:7b)

import csv
import os
import re
import subprocess
import sys
import time
import threading

MODEL = "mistral:7b"

def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        if MODEL not in result.stdout:
            subprocess.run(["ollama", "pull", MODEL])
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

my_books = {
    "books/1984_original.txt": "1984 by George Orwell",
    "books/hp_original.txt": "Harry Potter by J.K. Rowling"
}

def read_entire_book(file_path):
    if not os.path.exists(file_path):
        print(f" Файл не найден: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    size_kb = len(text) / 1024
    return text

def split_into_chunks(text, chunk_size=8000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def ask_ollama_for_chunk(chunk_text, book_name, chunk_num, total_chunks):
    prompt = f"""You are a literary scholar. Extract ONLY real cultural allusions from PART {chunk_num}/{total_chunks} of "{book_name}".

DEFINITION: A cultural allusion is an INDIRECT reference to:
- The Bible (Adam, Eve, Cain, Job, Moses, David, Goliath, Good Samaritan, Prodigal Son, etc.)
- Greek/Roman mythology (Achilles, Pandora, Sisyphus, Hercules, Atlas, Midas, Narcissus, etc.)
- Shakespeare (Hamlet, Romeo, Macbeth, Othello, King Lear, etc.)
- Other literature (Scrooge, Frankenstein, Jekyll and Hyde, Sherlock Holmes, etc.)
- History (Waterloo, Camelot, Napoleon, Churchill, etc.)
- Politics (Big Brother, Room 101, Doublethink, etc.)

DO NOT include:
- Ordinary conversation ("Hello", "How are you", "Good morning")
- Character names unless they are allusions to something else
- Descriptions of actions without cultural reference
- Common expressions ("Oh my God", "Thank you")

For each REAL allusion, write EXACTLY:
"phrase" -> what it refers to and its meaning

If you find no real allusions, say "NONE". Be strict.

Text part {chunk_num}:
{chunk_text}"""

    cmd = ["ollama", "run", MODEL]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=120)
        return stdout.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return ""
    except Exception as e:
        return ""

def parse_response(raw_response, book_name):
    results = []
    
    if not raw_response or raw_response.strip() == "NONE":
        return results
    
    pattern = r'"([^"]+)"\s*->\s*(.+?)(?=\n|$)'
    matches = re.findall(pattern, raw_response, re.MULTILINE)
    
    seen = set()
    for marker, definition in matches:
        marker = marker.strip()
        definition = definition.strip()
        
        if len(marker) < 3 or len(marker) > 80:
            continue
        if marker.lower() in ['the', 'a', 'an', 'and', 'of', 'to', 'in', 'for', 'on', 'with']:
            continue
        if marker in seen:
            continue
            
        seen.add(marker)
        results.append([marker, f"{definition} (from {book_name})"])
    
    return results

if __name__ == "__main__":    
    if not check_ollama():
        exit(1)
    
    all_allusions = []
    
    for filename, bookname in my_books.items():
        print(f"\n Книга: {filename}")
        full_text = read_entire_book(filename)
        
        if not full_text:
            continue
        
        chunks = split_into_chunks(full_text, chunk_size=8000)
        print(f"  Книга разбита на {len(chunks)} частей")
        
        for i, chunk in enumerate(chunks):
            print(f"Часть {i+1}/{len(chunks)}...")
            response = ask_ollama_for_chunk(chunk, bookname, i+1, len(chunks))
            allusions = parse_response(response, bookname)
            if allusions:
                print(f"Найдено: {len(allusions)} аллюзий")
            for marker, definition in allusions:
                all_allusions.append([marker, definition])
            
            time.sleep(0.5)
    
    if all_allusions:
        unique = {}
        for marker, definition in all_allusions:
            if marker not in unique:
                unique[marker] = definition
        
        with open("cultural_markers_dataset_2.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cultural_marker", "definition"])
            for marker, definition in sorted(unique.items()):
                writer.writerow([marker, definition])
        
        print(f"Всего аллюзий: {len(unique)}")
        
        print("\n Первые 20 аллюзий:")
        for i, (marker, definition) in enumerate(list(unique.items())[:20]):
            short_def = definition[:100] + "..." if len(definition) > 100 else definition
            print(f"{i+1}. {marker}")
            print(f"   → {short_def}")
            print()
    else:
        print("\n Не найдено ни одной аллюзии")