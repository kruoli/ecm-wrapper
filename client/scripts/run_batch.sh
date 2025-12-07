#!/bin/bash
echo "Starting ECM batch processing on $(wc -l < data/numbers.txt) numbers"
echo "=========================================="

count=0
total=$(wc -l < data/numbers.txt)

while IFS= read -r number <&3; do
    # Strip whitespace and carriage returns (handles Windows line endings)
    number=$(echo "$number" | tr -d '\r' | xargs)

    # Skip empty lines
    if [ -z "$number" ]; then
        continue
    fi

    count=$((count + 1))
    echo "[$count/$total] Starting ECM on: $number"
    python3 ecm_wrapper.py --composite "$number" --stage1-only -v --b1 11000000 --curves 1
    echo "[$count/$total] Completed"
    echo "---"
done 3< data/numbers.txt

echo "ECM batch processing complete!"