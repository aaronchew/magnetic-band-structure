#!/bin/bash

for file in $(cat listOfFiles.txt)
do cat $file | sed 's/*^/e/g' | sed 's/*I/j/g' > $(echo $file | sed 's/.csv/_mod.csv/g')
done