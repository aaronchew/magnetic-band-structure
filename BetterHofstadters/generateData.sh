#!/bin/bash

for i in $(seq 20)
do python3 generateData.py $i 20
done