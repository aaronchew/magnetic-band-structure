#!/bin/bash

for i in $(seq 10)
do for j in $(seq 13)
do python3 generateData6Pi.py $j $i
done done