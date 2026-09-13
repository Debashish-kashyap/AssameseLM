import re
from tqdm import tqdm

ASSAMESE_PATTERN = re.compile(r'[\u0980-\u09FF]')

def is_good_line(line):
    line = line.strip()
    
    if len(line) < 20:
        return False
    
    assamese_chars = len(ASSAMESE_PATTERN.findall(line))
    total_chars = len(line.replace(" ", ""))
    
    # less than 60% Assamese script → mixed/noisy line
    if total_chars == 0 or assamese_chars / total_chars < 0.6:
        return False
    
    return True

def clean_line(line):
    line = line.strip()
    # collapse multiple spaces
    line = re.sub(r' +', ' ', line)
    # remove zero-width chars
    line = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', line)
    return line

input_file  = "data/raw/sangraha_as.txt"
output_file = "data/processed/clean_as.txt"

kept = 0
dropped = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line in tqdm(fin, desc="Cleaning"):
        if is_good_line(line):
            fout.write(clean_line(line) + "\n")
            kept += 1
        else:
            dropped += 1

total = kept + dropped
print(f"Kept:    {kept:,}  ({100*kept/total:.1f}%)")
print(f"Dropped: {dropped:,}  ({100*dropped/total:.1f}%)")
print(f"Output:  data/processed/clean_as.txt")