#!/bin/bash

# Directories:
quantomDir="/Users/daniellersch/Desktop/SciDAC/ndata_paper/quantom-ips"
logDir="/Users/daniellersch/Desktop/SciDAC/ndata_paper"

# Settings to store / create the data:
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data"
nSamples=1000000
#epsilons=("1.0" "1.25" "0.5")
#coeffs=("[1.0,1.0]" "[-1.0,2.0]" "[-0.5,-0.5]")
epsilons=("1.0" "1.25" "0.25" "0.5" "0.75" "1.5")
coeffs=("[1.0,1.0]" "[-2.0,2.0]" "[-9.46,1.67]" "[4.20,0.74]" "[0.921,1.98]" "[-1.10,1.90]")

nSets=${#epsilons[@]}
for (( i=0; i<nSets; i++ )); do
   echo "Creating dataset $i"
   fullPath=$dataLoc"/dataset_$i.pkl"
   CMD=" $quantomDir/examples/gaussian_dataset_creation.py"
   CMD=$CMD" coefficients=${coeffs[$i]} epsilon=${epsilons[$i]} n_samples=$nSamples out_path=$fullPath"
   python $CMD
done