#!/bin/bash
mkdir -p build
cd build
cmake -S ../ -B ./
make && ./lil_name_gen
cd ..