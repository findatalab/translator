# Выявление аллюзий (модель: Qwen2.5:7b)

import csv
import os
import re
import subprocess
import sys
import time
import threading

MODEL = "qwen2.5:7b"

def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        if MODEL not in result.stdout:
            subprocess.run(["ollama", "pull", MODEL])
        return True
    except Exception as e:
        print(f"Ошибка {e}")
        return False

my_books = {
    "books/1984_original.txt": "1984 by George Orwell",
    "books/hp_original.txt": "Harry Potter by J.K. Rowling"
}

def read_entire_book(file_path):
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    size_kb = len(text) / 1024
    return text

def split_into_chunks(text, chunk_size=10000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def ask_ollama_for_chunk(chunk_text, book_name, chunk_num, total_chunks):
    prompt = f"""You are a literary scholar. Extract ALL cultural allusions from this PART {chunk_num}/{total_chunks} of "{book_name}".

DEFINITION: An allusion is an indirect reference to the Bible, mythology, history, Shakespeare, other literature, famous people, or political concepts.

EXAMPLES:
- "Big Brother" -> totalitarian surveillance from Orwell's 1984
- "Room 101" -> torture chamber with worst fears
- "Muggle" -> non-magical person in Harry Potter
- "Garden of Eden" -> biblical paradise lost
- "Achilles' heel" -> vulnerable spot from Greek myth
- "Scrooge" -> miserly person from Dickens

For each allusion you find, write EXACTLY:
"phrase" -> explanation

If you find many allusions, list them all. Be thorough.

Text part {chunk_num}:
{chunk_text}

List all allusions in this part:"""

    cmd = ["ollama", "run", MODEL]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=180)
        return stdout.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return ""
    except Exception as e:
        return ""

def parse_response(raw_response, book_name, chunk_num):
    results = []
    
    if not raw_response:
        return results
    
    pattern = r'"([^"]+)"\s*->\s*(.+?)(?=\n|$)'
    matches = re.findall(pattern, raw_response, re.MULTILINE)
    
    for marker, definition in matches:
        marker = marker.strip()
        definition = definition.strip()
        
        if len(marker) > 2 and len(marker) < 100 and len(definition) > 5:
            if marker.lower() not in ['the', 'a', 'an', 'and', 'of', 'to', 'in', 'for']:
                results.append([marker, f"{definition} (from {book_name})"])
    
    return results

if __name__ == "__main__":
    if not check_ollama():
        exit(1)
    
    all_allusions = []
    
    for filename, bookname in my_books.items():
        print(f"Книга: {filename}")
        full_text = read_entire_book(filename)
        
        if not full_text:
            continue
        
        chunks = split_into_chunks(full_text, chunk_size=10000)
        print(f"Книга разбита на {len(chunks)} частей")
        
        book_allusions = []
        
        for i, chunk in enumerate(chunks):
            print(f"Часть {i+1}/{len(chunks)}...")
            
            response = ask_ollama_for_chunk(chunk, bookname, i+1, len(chunks))
            
            allusions = parse_response(response, bookname, i+1)
            print(f"Найдено: {len(allusions)} аллюзий")
            
            book_allusions.extend(allusions)
            
            time.sleep(1)
        
        print(f"\nВсего в {bookname}: {len(book_allusions)} аллюзий")
        all_allusions.extend(book_allusions)
    
    if all_allusions:
        unique = {}
        for marker, definition in all_allusions:
            if marker not in unique:
                unique[marker] = definition
        
        sorted_unique = sorted(unique.items(), key=lambda x: x[0])
        
        with open("cultural_markers_dataset.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cultural_marker", "definition"])
            for marker, definition in sorted_unique:
                writer.writerow([marker, definition])
        
        print(f"Всего было найдено уникальных аллюзий: {len(unique)}")
        
        print("\n Первые 20 аллюзий:")
        for i, (marker, definition) in enumerate(sorted_unique[:20]):
            short_def = definition[:100] + "..." if len(definition) > 100 else definition
            print(f"{i+1}. {marker}")
            print(f"   → {short_def}")
            print()
    else:
        print("\n Не найдено ни одной аллюзии")