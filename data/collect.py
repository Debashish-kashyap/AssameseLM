from datasets import load_dataset
from tqdm import tqdm
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading Sangraha Verified Assamese...")
sangraha = load_dataset("ai4bharat/sangraha", data_dir="verified/asm", split="train")

with open("data/raw/sangraha_as.txt", "w", encoding="utf-8") as f:
    for item in tqdm(sangraha):
        line = item["text"].strip()
        if line:
            f.write(line + "\n")

print(f"Sangraha done: {len(sangraha)} docs")

print("Downloading IndicCorpV2 Assamese...")
indicorp = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="asm_Beng")

with open("data/raw/indiccorp_as.txt", "w", encoding="utf-8") as f:
    for item in tqdm(indicorp):
        line = item["text"].strip()
        if line:
            f.write(line + "\n")

print(f"IndicCorpV2 done: {len(indicorp)} lines")

print("Downloading Assamese Wikipedia...")
wiki = load_dataset("wikipedia", "20231101.as", split="train")

with open("data/raw/wikipedia_as.txt", "w", encoding="utf-8") as f:
    for item in tqdm(wiki):
        text = item["text"].strip()
        if text:
            f.write(text + "\n")

print(f"Wikipedia done: {len(wiki)} articles")