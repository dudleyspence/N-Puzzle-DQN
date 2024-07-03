#!/bin/bash

# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Setup completed. Activate the virtual environment with 'source env/bin/activate'."
