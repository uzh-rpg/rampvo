#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <destination_directory>"
    echo "Set UNZIP_FILES to true to unzip the downloaded files."
    echo "Set DELETE_FILES to true to delete the downloaded zip files after unzipping."
    exit 1
fi

DEST_DIR="$1"
ROOT_URL="https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"
FILES=(
    "abandonedfactory_night.zip"
    "abandonedfactory.zip"
    "amusement.zip"
    "carwelding.zip"
    "endofworld.zip"
    "gascola.zip"
    "hospital.zip"
    "japanesealley.zip"
    "neighborhood.zip"
    "ocean.zip"
    "office2.zip"
    "office.zip"
    "oldtown.zip"
    "seasidetown.zip"
    "seasonsforest_winter.zip"
    "seasonsforest.zip"
    "soulcity.zip"
    "westerndesert.zip"
)

mkdir -p "$DEST_DIR"

for file in "${FILES[@]}"; do
    echo "Processing $file..."

    DOWNLOAD_URL="${ROOT_URL}/${file}"
    TARGET_FILE="$DEST_DIR/$file"

    echo "Downloading $DOWNLOAD_URL..."
    curl -o "$TARGET_FILE" "$DOWNLOAD_URL"

    # Check if UNZIP_FILES variable is set and unzip if true
    if [ "$UNZIP_FILES" ]; then
        BASE_NAME=$(basename "$file" .zip)
        UNZIP_DIR="$DEST_DIR/$BASE_NAME"

        mkdir -p "$UNZIP_DIR"

        echo "Unzipping $TARGET_FILE to $UNZIP_DIR..."
        unzip -o "$TARGET_FILE" -d "$UNZIP_DIR"

        # Check if DELETE_FILES variable is set and delete the zip file if true
        if [ "$DELETE_FILES" ]; then
            echo "Deleting $TARGET_FILE..."
            rm -f "$TARGET_FILE"
        fi
    fi
done

echo "All files processed."
