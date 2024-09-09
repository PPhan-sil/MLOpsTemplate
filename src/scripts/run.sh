#!/usr/bin/env bash
python train.py --yaml_file exp/experiment_1.yml
python train.py --yaml_file exp/experiment_2.yml
python eval.py --epochs 5 --trials 3
