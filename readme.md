# Install requirements
pip install -r dt_coursework/requirements.txt

# Run full pipeline on both datasets with figures (10-fold + nested pruning)
python dt_coursework/run.py --clean --noisy --k 10 --make-figures
