# Generate karel datasets
python karel_env/generator.py
# Append unseen demonstrations to each programs
python karel_env/append_demonstration.py
# Add perception primitives to each demonstrations
python karel_env/add_per.py
