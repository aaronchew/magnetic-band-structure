#!/bin/bash

for i in $(seq 20)
do for j in $(seq 20)
do echo $i" "$j | /Applications/Mathematica.app/Contents/MacOS/MathKernel -noprompt -run "<<CreateFormFactors.m"
done
done