#!/bin/bash

source venv/bin/activate

## First, run demo.py to show manual rules and how they work.
python demo.py

## Show rules 
python -m fuzzylogic_cartpole.visualize_fuzzy_sets manual_config.yaml

## Second run fuzzy rule optimization
python -m fuzzylogic_cartpole.optimize_rules_ea 

## Show performance of optimized rules
python demo.py --config optmized_rules.yaml

## Third run fuzzy set optimization
python -m fuzzy_logic_cartpole.optimize_fuzzy_sets_ea 

## Show performance of optimized fuzzy sets
python demo.py --config optimized_fuzzy_sets.yaml

## Show optmized fuzzy sets
python -m fuzzylogic_cartpole.visualize_fuzzy_sets python demo.py --config optimized_fuzzy_sets.yaml