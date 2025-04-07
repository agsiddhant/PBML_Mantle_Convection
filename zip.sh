#!/bin/bash

# Root folder and output zip
MAIN_DIR="TPH"
ZIP_FILE="TPH_filtered.zip"

# Clean up old stuff
rm -f "$ZIP_FILE" file_list.txt

# Find matching files with full relative path
find "$MAIN_DIR" -type f \( -name "e1_*select_snaps*" -o ! -name "e1_*" \) > file_list.txt

# Double-check: file_list.txt should contain lines like:
# TPH/train/sim_95/e1_vprev_data_select_snaps.pt

# Now zip everything based on that list
zip -r "$ZIP_FILE" -@ < file_list.txt

# Cleanup
#rm file_list.txt

echo "✅ Done: created $ZIP_FILE with preserved folder structure."
