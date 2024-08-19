#!/bin/bash

# Switch to project root
cd ..

# Setup virtual environment
virtualenv .env -p python3.11
source .env/bin/activate
pip install --upgrade pip

# Install PyTorch
pip install --index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
