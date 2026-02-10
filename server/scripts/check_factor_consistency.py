#!/usr/bin/env python3
"""
Check factor consistency across the database.

For each composite that has factors recorded, verify that no recorded factor
still divides current_composite. If it does, the factor wasn't properly
divided out (e.g. the composite was reset by an external sync).

Usage:
    python3 scripts/check_factor_consistency.py

    # With custom database URL
    DATABASE_URL="postgresql://..." python3 scripts/check_factor_consistency.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.config import get_settings


def main():
    settings = get_settings()
    database_url = os.environ.get("DATABASE_URL", settings.database_url)
    engine = create_engine(database_url)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.id, c.current_composite, c.is_fully_factored, c.is_complete,
                   c.digit_length,
                   f.id as factor_id, f.factor
            FROM composites c
            JOIN factors f ON f.composite_id = c.id
            ORDER BY c.id, f.id
        """)).fetchall()

    if not rows:
        print("No composites with factors found.")
        return

    # Group by composite
    composites = {}
    for row in rows:
        cid = row[0]
        if cid not in composites:
            composites[cid] = {
                'id': cid,
                'current_composite': row[1],
                'is_fully_factored': row[2],
                'is_complete': row[3],
                'digit_length': row[4],
                'factors': []
            }
        composites[cid]['factors'].append({
            'factor_id': row[5],
            'factor': row[6]
        })

    print(f"Checking {len(composites)} composites with factors...\n")

    issues = []

    for cid, comp in composites.items():
        current = int(comp['current_composite'])

        for finfo in comp['factors']:
            factor = int(finfo['factor'])

            if current % factor == 0:
                issues.append({
                    'composite_id': cid,
                    'factor_id': finfo['factor_id'],
                    'factor': finfo['factor'],
                    'current_prefix': comp['current_composite'][:30],
                    'is_fully_factored': comp['is_fully_factored'],
                    'is_complete': comp['is_complete'],
                    'digit_length': comp['digit_length'],
                })

    if issues:
        print("=" * 70)
        print(f"WARNING: {len(issues)} factor(s) still divide current_composite!")
        print("These composites may have been reset and need factor division re-applied.")
        print("=" * 70)
        for item in issues:
            print(f"  Composite {item['composite_id']} ({item['digit_length']} digits): "
                  f"factor {item['factor']} still divides current_composite "
                  f"(fully_factored={item['is_fully_factored']}, complete={item['is_complete']})")
    else:
        print("All factors are consistent. No recorded factor divides its current_composite.")


if __name__ == "__main__":
    main()
