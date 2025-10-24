import json
import os
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
DATA_FOLDER = "data"  # Folder containing your JSON files (1.json, 2.json, ...)
OUTPUT_FILE = "quran_with_embeddings.json"  # Combined output

# Load model once
print("=" * 60)
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print(f"Model '{MODEL_NAME}' loaded successfully.")
print("=" * 60)


def get_embedding(text):
    """Generate embeddings locally using SentenceTransformer"""
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"  ✗ Error encoding text: {e}")
        return None


def process_verses():
    """Process all Quran verses and generate embeddings locally"""
    all_surahs = []
    total_verses = 0
    processed_verses = 0

    # Count total verses
    print("Counting verses...")
    for i in range(1, 115): 
        json_file = Path(DATA_FOLDER) / f"{i}.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                surah_data = json.load(f)
                total_verses += len(surah_data['verses'])
    print(f"Total verses to process: {total_verses}")
    print("=" * 60)

    # Process each surah
    for i in range(1, 115):
        json_file = Path(DATA_FOLDER) / f"{i}.json"
        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            surah_data = json.load(f)

        print(f"\nProcessing Surah {i}: {surah_data['name']} ({surah_data['transliteration']})")
        print(f"  Verses: {len(surah_data['verses'])}")

        # Process each verse
        for verse in surah_data['verses']:
            verse_text = verse["translation"]
            embedding = get_embedding(verse_text)

            if embedding is not None:
                verse["embedding"] = embedding
                processed_verses += 1
                progress = (processed_verses / total_verses) * 100
                print(f"  ✓ Verse {verse['id']} - Progress: {processed_verses}/{total_verses} ({progress:.1f}%)")
            else:
                verse["embedding"] = None
                print(f"  ✗ Failed to embed verse {verse['id']}")

        all_surahs.append(surah_data)
        print(f"  Completed Surah {i}")

    # Save all data to one JSON file 
    print("\n" + "=" * 60)
    print(f"Saving all data to {OUTPUT_FILE}...")
    output_data = {
        "total_surahs": len(all_surahs),
        "total_verses": total_verses,
        "surahs": all_surahs
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✓ Saved {processed_verses} verses with embeddings")
    print(f"✓ Output file size: {file_size:.2f} MB")
    print("=" * 60)
    return output_data


def verify_embeddings(output_file):
    """Verify embedding structure"""
    print("\nVerifying embeddings...")
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_verses = 0
    verses_with_embeddings = 0
    for surah in data["surahs"]:
        for verse in surah["verses"]:
            total_verses += 1
            if verse.get("embedding") is not None:
                verses_with_embeddings += 1

    print(f"Total verses: {total_verses}")
    print(f"Verses with embeddings: {verses_with_embeddings}")
    print(f"Success rate: {(verses_with_embeddings / total_verses) * 100:.1f}%")

    # Inspect one example
    sample = data["surahs"][0]["verses"][0].get("embedding")
    if sample:
        print(f"Embedding dimension: {len(sample)}")
        print(f"Sample (first 5 values): {sample[:5]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Quran Verse Embedding Generator (Local Mode)")
    print("=" * 60)

    if not os.path.exists(DATA_FOLDER):
        print(f"ERROR: Data folder '{DATA_FOLDER}' not found!")
        exit(1)

    start = time.time()
    process_verses()
    verify_embeddings(OUTPUT_FILE)
    end = time.time()

    print(f"\nTotal processing time: {(end - start) / 60:.1f} minutes")
    print("=" * 60)
    print("✅ Done! You can now use 'quran_with_embeddings.json' in your web app.")
