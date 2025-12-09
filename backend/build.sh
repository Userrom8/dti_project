#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a directory for the model (optional, keeps things clean)
mkdir -p saved_models

# 3. Download the model from GitHub Releases
# REPLACE THE URL below with your specific GitHub Release asset URL
# Note: Use -L to follow redirects (GitHub uses them for downloads)
curl -L -o saved_models/best_checkpoint.pt "https://github.com/Userrom8/dti_project/releases/download/v1.0-model/best_checkpoint.pt"

echo "Build finished. Model downloaded to saved_models/best_checkpoint.pt"