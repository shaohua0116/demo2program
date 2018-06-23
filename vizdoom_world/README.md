# Vizdoom Environment
This directory includes code for Vizdoom environments, which includes:
- Random programs and demonstration generator
- Domain specific language interpreter

## Dependencies
Using vizdoom environment requires [VizDoom Deterministic](https://github.com/HyeonwooNoh/ViZDoomDeterministic) as a dependency.
You can install the VizDoom Deterministic as follows:
```bash
git clone https://github.com/HyeonwooNoh/ViZDoomDeterministic
cd ViZDoomDeterministic
# use pip3 for Python3
sudo pip install .
```
Please find the detailed installation instruction of the VizDoom Deterministic from [this repository](https://github.com/HyeonwooNoh/ViZDoomDeterministic).

## Dataset generation

Datasets used in the paper (vizdoom_dataset, vizdoom_dataset_ifelse) are generated with the following script
```bash
./vizdoom_world/generate_dataset.sh
```

## Domain specific language
The interpreter and random program generator for vizdoom domain specific language (DSL) is in the [dsl directory](./dsl). You can find detailed definition of the DSL from the supplementary material of the paper.
