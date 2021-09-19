#!/bin/bash

for i in $(seq 20)
do for j in $(seq $i)
do python3 generateData.py $j $i
done done
