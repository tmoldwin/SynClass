# Run contrastive learning with different Z-slices from same synapse as view pair.
# Same settings as the last EM-only run: 5k samples, 30 epochs.
python synapse_contrastive.py --epochs 30 --batch_size 64 --max_samples 5000 --checkpoint_every 10 --multi_slice
