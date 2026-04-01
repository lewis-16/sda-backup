#!/bin/bash

DEST_DIR="/media/ubuntu/sda/Monkey/TVSD/monkeyF/20240118"
LINK_FILE="/media/ubuntu/sda/Monkey/link.txt"

mkdir -p "$DEST_DIR"

while IFS= read -r url; do
    if [ -n "$url" ]; then
        echo "Downloading: $url"
        wget -c -O "$DEST_DIR/$(basename "$url")" "$url"
    fi
done < "$LINK_FILE"
