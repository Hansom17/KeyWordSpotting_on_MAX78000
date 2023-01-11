#!/bin/sh
python train.py --epochs 50 --optimizer Adam --lr 0.000000001 --wd 0 --deterministic --compress policies/schedule_kws20_v2.yaml --exp-load-weights-from qat_checkpoint.pth.tar --model ai85kws20netv2 --dataset mltd --confusion --device MAX78000 "$@"
