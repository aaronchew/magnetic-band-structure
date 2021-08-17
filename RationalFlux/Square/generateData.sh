#!/bin/bash

for i in $(seq 12)
do python3 generateData.py $i 12
done