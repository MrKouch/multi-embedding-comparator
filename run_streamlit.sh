#!/bin/bash
source /shared/mrkouch/miniconda3/etc/profile.d/conda.sh
conda activate /shared/mrkouch/envs/mec
cd /shared/mrkouch/multi-embedding-comparator

# Check if port 8501 is in use
if lsof -i :8501 &>/dev/null; then
  echo "⚠️  Port 8501 is already in use. Please free it or choose a different one."
  echo "You can check with: lsof -i :8501"
  exit 1
fi

streamlit run app.py --server.port 8506


