#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A800:2
#SBATCH --time=02:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=./ana_mg_v0.log 
#SBATCH --job-name=MG
# This line is very important --> Make shell aware of your conda env
source /etc/profile.d/conda.sh 
# Activate your conda env
conda activate quantom_env


# Directories:
quantomDir="/work/data_science/dlersch/SciDAC/mdata_paper/quantom-ips"
logDir="/work/data_science/dlersch/SciDAC/mdata_paper"
dataLoc="/work/data_scsience/quantom/data/mgaussian_sets"
# Training:
nEpochs=5000
readFreq=25
printFreq=1000
snapFreq=250
outerFreq=2
genLR=1e-4
discLR=1e-4
nRanks=2
gradMode="conv_arar"
nSets=2
version=2
batchSize=1000
# Data
nSamples=100000
pathList=()
for ((i=0; i<nSets; i++)); do
    pathList+=("$dataLoc/dataset_${i}.pkl")
done
printf -v paths "%s," "${pathList[@]}"
paths="[${paths%,}]"

# Results:
resultLoc="$logDir/runs/results_ifarm_mgaussian_"$nSets"sets_N$nRanks"_v$version

# Define the basic command line:
CMD="python $quantomDir/examples/distributed_mgaussian_workflow.py"
# Set optimizer parameters:
CMD=$CMD" opt.batch_size=$batchSize"
CMD=$CMD" opt.n_epochs=$nEpochs opt.frequency=$readFreq opt.print_frequency=$printFreq"
CMD=$CMD" opt.snapshot_frequency=$snapFreq opt.outer_group_update_frequency=$outerFreq"
CMD=$CMD" opt.gradient_transport.gradient_sync_mode=$gradMode"
CMD=$CMD" hydra.run.dir=$resultLoc"
#CMD=$CMD" env.average=True"
# Data parser:
CMD=$CMD" dataloader/dataset=gaussian_pickles dataloader.dataset.paths=$paths dataloader.dataset.n_samples=$nSamples"
#CMD=$CMD" dataloader.dataset.n_gaussians=$nGaussians"
#CMD=$CMD" dataloader.dataset.samples_per_batch=$nSamples dataloader.dataset.samples_per_dataset=$nEvents"
# GAN:
#CMD=$CMD" opt.generator.hidden_dims=[128,128,128,128] opt.discriminator.hidden_dims=[128,128,128,128]"
CMD=$CMD" opt.generator.hidden_weight_initializer=kaiming_uniform opt.generator.hidden_bias_initializer=normal"
CMD=$CMD" opt.generator.out_weight_initializer=xavier_uniform opt.generator.out_bias_initializer=normal"
CMD=$CMD" opt.discriminator.hidden_weight_initializer=kaiming_uniform opt.discriminator.hidden_bias_initializer=normal"
CMD=$CMD" opt.discriminator.out_weight_initializer=xavier_uniform opt.discriminator.out_bias_initializer=normal"

mpirun --mca btl_tcp_port_min_v4 32768 --mca btl_tcp_port_range_v4 28230 $CMD
