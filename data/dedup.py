print("Deduplicating...")
with open("data/processed/clean_as.txt", encoding="utf-8") as f:
    lines = f.readlines()

seen = set()
unique_lines = []
for line in lines:
    if line not in seen:
        seen.add(line)
        unique_lines.append(line)

with open("data/processed/clean_dedup_as.txt", "w", encoding="utf-8") as f:
    f.writelines(unique_lines)

print(f"Before: {len(lines):,}")
print(f"After:  {len(unique_lines):,}")
print(f"Removed: {len(lines)-len(unique_lines):,}")