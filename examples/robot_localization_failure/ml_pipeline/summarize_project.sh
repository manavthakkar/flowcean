#!/usr/bin/env bash

# ============================================================
# summarize_project.sh
# Generates:
#   - structure.txt  : folder hierarchy
#   - code_dump.txt  : concatenated source code
#   - project.zip    : full archive of project
# ============================================================

# -------- CONFIGURE FILE TYPES TO INCLUDE ----
EXTENSIONS=("py" "md" "json" "yaml" "yml" "txt" "ini" "sh")  # add more if needed

IGNORED_DIRS="env|venv|__pycache__|node_modules|.git|.mypy_cache|.pytest_cache|build|dist"

# --------------------- 1. Project Structure ------------------
echo "[1/3] Generating directory tree..."
tree -I "$IGNORED_DIRS" > structure.txt
echo "✓ structure.txt created"

# ---------------------- 2. Code Dump -------------------------
echo "[2/3] Dumping source code..."

> code_dump.txt  # empty file

for EXT in "${EXTENSIONS[@]}"; do
    FILES=$(find . -type f -name "*.$EXT" | sort)

    if [ -n "$FILES" ]; then
        for FILE in $FILES; do
            echo -e "\n\n==================== FILE: $FILE ====================\n" >> code_dump.txt
            cat "$FILE" >> code_dump.txt
        done
    fi
done

echo "✓ code_dump.txt created"

# ----------------------- 3. ZIP Archive ----------------------
echo "[3/3] Creating ZIP archive..."
zip -r project.zip . -x "*env/*" "*venv/*" "*__pycache__/*" "*node_modules/*" "*.git/*"

echo "✓ project.zip created"

echo
echo "🎉 Done!"
