#!/bin/bash

for nLayers in 1 3 5; do
	for nBlocks in 1 3 5 7; do
		for betaMult in 10 30 90 270; do
			python3 trn.py --nLayers "$nLayers" --nBlocks "$nBlocks" --betaMult "$betaMult" --testExam Exam9992Test --testTimepoint 0;
		done;
	done;
done
