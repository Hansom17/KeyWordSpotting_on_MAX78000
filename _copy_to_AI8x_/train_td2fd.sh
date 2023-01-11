#!/bin/sh
python train.py --epochs 50 --optimizer Adam --lr 0.0001 --wd 0 --deterministic --compress policies/schedule.yaml --model td2fd --dataset mltdtofd --device MAX78000 "$@"
