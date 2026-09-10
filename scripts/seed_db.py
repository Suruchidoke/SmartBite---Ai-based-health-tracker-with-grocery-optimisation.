"""
SmartBite Dataset Seeder Script
================================
Seeds MongoDB collections (nutrition, food, grocery, recipes) from local CSV datasets.
Run this script once during deployment or testing:
    python scripts/seed_db.py
"""

import os
import sys
import argparse
import pandas as pd
from pymongo import MongoClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import Config

DATASET_DIR = Config.DATASET_DIR
MONGO_URI = Config.MONGO_URI
DB_NAME = Config.DB_NAME


def seed_database(force=False):
    print(f"Connecting to MongoDB at {MONGO_URI} (DB: {DB_NAME})...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client[DB_NAME]

    collection_files = {
        "nutrition": ["CSV_VERSION.csv", "nutrition.csv"],
        "food": ["indian_food.csv", "food.csv"],
        "grocery": ["grocery_dataset.csv", "openfoodfacts_grocery.csv", "Grocery_data (1).csv"],
        "recipes": ["recipes_dataset.csv", "recipe_dataset.csv", "recipes.csv"],
    }

    for col_name, file_candidates in collection_files.items():
        col = db[col_name]
        existing = col.count_documents({})
        if existing > 0 and not force:
            print(f"Collection '{col_name}' already contains {existing} records. Skipping (use --force to reseed).")
            continue

        if force and existing > 0:
            print(f"Clearing collection '{col_name}'...")
            col.delete_many({})

        # Find first matching CSV
        loaded = False
        for fname in file_candidates:
            path = os.path.join(DATASET_DIR, fname)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, on_bad_lines="skip")
                    if not df.empty:
                        # Convert to dict and insert
                        records = df.to_dict("records")
                        # Batch insert in chunks of 5000
                        chunk_size = 5000
                        total_inserted = 0
                        for i in range(0, len(records), chunk_size):
                            chunk = records[i : i + chunk_size]
                            res = col.insert_many(chunk)
                            total_inserted += len(res.inserted_ids)
                        print(f"[OK] Seeded '{col_name}' with {total_inserted} records from '{fname}'.")
                        loaded = True
                        break
                except Exception as e:
                    print(f"[WARN] Error reading {path}: {e}")

        if not loaded:
            print(f"[INFO] No CSV found for collection '{col_name}'.")

    print("\nDatabase seeding completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SmartBite MongoDB collections from CSV datasets.")
    parser.add_argument("--force", action="store_true", help="Drop and re-seed collections even if they contain data.")
    args = parser.parse_args()
    seed_database(force=args.force)
