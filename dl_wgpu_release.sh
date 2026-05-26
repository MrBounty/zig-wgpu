#!/bin/bash

# 1. SET THIS to the exact git tag of the release you are downloading
# (e.g., "v22.1.0.5" or "v0.19.4.1")
VERSION="v29.0.0.0" 
BASE_URL="https://github.com/gfx-rs/wgpu-native/releases/download/$VERSION"

# Move into the wgpu-native folder
cd libs/wgpu-native || { echo "Could not find libs/wgpu-native"; exit 1; }

# Map the zip filenames to your target directory structure
declare -A TARGETS=(
    ["wgpu-linux-x86_64-release.zip"]="x86_64-linux"
    ["wgpu-linux-aarch64-release.zip"]="aarch64-linux"
    ["wgpu-macos-x86_64-release.zip"]="x86_64-macos"
    ["wgpu-macos-aarch64-release.zip"]="aarch64-macos"
    ["wgpu-windows-x86_64-msvc-release.zip"]="x86_64-windows"
    ["wgpu-windows-aarch64-msvc-release.zip"]="aarch64-windows"
)

for ZIP in "${!TARGETS[@]}"; do
    DIR="${TARGETS[$ZIP]}"
    echo "Processing $DIR..."
    
    # Create the target directory
    mkdir -p "$DIR"
    
    # Download the zip file
    if wget -q --show-progress "$BASE_URL/$ZIP" -O "$ZIP"; then
        # Extract directly into the target directory
        unzip -q -o "$ZIP" -d "$DIR"
        echo "Successfully extracted to $DIR/"
    else
        echo "Failed to download $ZIP. Check your VERSION tag."
    fi
    
    # Remove the downloaded zip to keep things clean
    rm -f "$ZIP"
done

echo "Done! Your directories are set up."
