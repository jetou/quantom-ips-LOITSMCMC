#!/bin/bash

# Directories:
quantomDir="/Users/daniellersch/Desktop/SciDAC/ndata_paper/quantom-ips"
logDir="/Users/daniellersch/Desktop/SciDAC/ndata_paper"
dataLoc="/Users/daniellersch/Desktop/SciDAC/sample_data/mgaussian_sets"
# Training:
nEpochs=5000
readFreq=25
printFreq=1000
snapFreq=250
outerFreq=2
genLR=1e-5
discLR=1e-4
nRanks=2
gradMode="conv_arar"
gradTransport="ARAR"
nSets=2
version=2
batchSize=100
# Data
nSamples=10000
pathList=()
for ((i=0; i<nSets; i++)); do
    pathList+=("$dataLoc/dataset_${i}.pkl")
done
printf -v paths "%s," "${pathList[@]}"
paths="[${paths%,}]"

# Results:
resultLoc="$logDir/results_local_mgaussian_"$nSets"sets_N$nRanks"_v$version

# Define the basic command line:
CMD="python $quantomDir/examples/distributed_mgaussian_workflow.py"
# Set optimizer parameters:
CMD=$CMD" opt.batch_size=$batchSize gradient_transport@opt.gradient_transport=$gradTransport"
CMD=$CMD" opt.n_epochs=$nEpochs opt.frequency=$readFreq opt.print_frequency=$printFreq"
CMD=$CMD" opt.snapshot_frequency=$snapFreq opt.outer_group_update_frequency=$outerFreq"
CMD=$CMD" opt.gradient_transport.gradient_sync_mode=$gradMode"
CMD=$CMD" hydra.run.dir=$resultLoc"
# Data parser:
CMD=$CMD" dataloader/dataset=gaussian_pickles dataloader.dataset.paths=$paths dataloader.dataset.n_samples=$nSamples"
# GAN:
CMD=$CMD" opt.generator.hidden_dims=[128,128,128,128] opt.discriminator.hidden_dims=[128,128,128,128]"
CMD=$CMD" opt.generator.hidden_weight_initializer=kaiming_uniform opt.generator.hidden_bias_initializer=normal"
CMD=$CMD" opt.generator.out_weight_initializer=xavier_uniform opt.generator.out_bias_initializer=normal"
CMD=$CMD" opt.discriminator.hidden_weight_initializer=kaiming_uniform opt.discriminator.hidden_bias_initializer=normal"
CMD=$CMD" opt.discriminator.out_weight_initializer=xavier_uniform opt.discriminator.out_bias_initializer=normal"

mpirun -n $nRanks $CMD