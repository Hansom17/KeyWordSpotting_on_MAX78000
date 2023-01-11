#!/bin/sh
python train.py --epochs 50 --optimizer Adam --lr 0.0001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model kwscnn --dataset mlcommos --confusion --device MAX78000 --use-bias "$@"
