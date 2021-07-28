#!/bin/bash

for i in $(seq 4)
do for j in $(seq 12)
do python3 Testme.py $i $j
done
done