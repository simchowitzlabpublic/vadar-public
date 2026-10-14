#!/bin/bash
# All 4 Adroit envs with seed=1. Copy individual sbatch lines and fire from terminal.

sbatch scripts/ogpo/adroit.sh --seed=1 --env_name=AdroitHandDoor-v1     --run_name=ogpo_adroit_door     --wandb_name=ogpo_adroit_door
sbatch scripts/ogpo/adroit.sh --seed=1 --env_name=AdroitHandHammer-v1   --run_name=ogpo_adroit_hammer   --wandb_name=ogpo_adroit_hammer
sbatch scripts/ogpo/adroit.sh --seed=1 --env_name=AdroitHandPen-v1      --run_name=ogpo_adroit_pen      --wandb_name=ogpo_adroit_pen
sbatch scripts/ogpo/adroit.sh --seed=1 --env_name=AdroitHandRelocate-v1 --run_name=ogpo_adroit_relocate --wandb_name=ogpo_adroit_relocate
