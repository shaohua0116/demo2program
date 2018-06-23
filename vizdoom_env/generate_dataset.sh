# Generate small vizdoom datasets with different seeds
# Note that we first generate small datasets to make generation parallelizable.
# We generate datasets with 40 seen demonstrations for generalization experiments,
# but we used only 25 seen demonstrations for training as described in the paper.
python vizdoom_world/generator.py --max_demo_length 8 --seed 123
python vizdoom_world/generator.py --max_demo_length 8 --seed 234
python vizdoom_world/generator.py --max_demo_length 8 --seed 345
python vizdoom_world/generator.py --max_demo_length 8 --seed 456
python vizdoom_world/generator.py --max_demo_length 8 --seed 567
python vizdoom_world/generator.py --max_demo_length 8 --seed 678
python vizdoom_world/generator.py --max_demo_length 8 --seed 789
python vizdoom_world/generator.py --max_demo_length 8 --seed 890
python vizdoom_world/generator.py --max_demo_length 20 --seed 234
python vizdoom_world/generator.py --max_demo_length 20 --seed 789
# Merge datasets
python vizdoom_world/merge_datasets.py --dataset_paths\
    datasets/vizdoom_small_len8_seed123 \
    datasets/vizdoom_small_len8_seed234 \
    datasets/vizdoom_small_len8_seed345 \
    datasets/vizdoom_small_len8_seed456 \
    datasets/vizdoom_small_len8_seed567 \
    datasets/vizdoom_small_len8_seed678 \
    datasets/vizdoom_small_len8_seed789 \
    datasets/vizdoom_small_len8_seed890 \
    datasets/vizdoom_small_len20_seed234 \
    datasets/vizdoom_small_len20_seed789

# For if-else experiments, you can generate datasets with the following script
python vizdoom_word/generator_ifelse.py
