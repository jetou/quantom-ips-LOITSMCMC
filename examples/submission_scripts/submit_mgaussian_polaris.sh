#!/bin/bash -l
#PBS -l select=10:ncpus=4:ngpus=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -A EE-ECP

# This script lets you run the workflow on Polaris 

# Set your environment:                                                                                                                                                                                                                                    
module use /soft/modulefiles
module load conda
source /home/dlersch/qm_mdata_env/bin/activate

# Directories:
quantomDir="/home/dlersch/SCUDAC/mdata_paper/quantom-ips"
driver="$quantomDir/examples/distributed_mgaussian_workflow.py"
logDir="/grand/EE-ECP/quantom_scidac/results"
dataLoc="/grand/EE-ECP/quantom_scidac/sample_data/mgaussian_sets"
# Computing resources:
nSets=1
ensembleSize=10
multiPlicity=1
nNodes=$((nSets * multiPlicity))
nRanks=$((nNodes * 4))
reduction=$((multiPlicity * 4))
echo "Using $nRanks ranks" 
offset=0
step=$((nNodes - 1))

# Training:
nEpochs=10000
readFreq=50
printFreq=2000
snapFreq=500
outerFreq=2
genLR=1e-5
discLR=1e-4
gradMode="conv_arar"
gradTransport="ARAR"
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
resultLocBase="$logDir/mdata_runs/results_polaris_mgaussian_"$nSets"sets_N$nRanks"

# Set optimizer parameters:
CMD=" opt.batch_size=$batchSize gradient_transport@opt.gradient_transport=$gradTransport"
CMD=$CMD" opt.n_epochs=$nEpochs opt.frequency=$readFreq opt.print_frequency=$printFreq"
CMD=$CMD" opt.snapshot_frequency=$snapFreq opt.outer_group_update_frequency=$outerFreq"
CMD=$CMD" opt.gradient_transport.gradient_sync_mode=$gradMode"
# Data parser:
CMD=$CMD" dataloader/dataset=gaussian_pickles dataloader.dataset.paths=$paths dataloader.dataset.n_samples=$nSamples"
# GAN:
CMD=$CMD" opt.generator.hidden_dims=[128,128,128,128] opt.discriminator.hidden_dims=[128,128,128,128]"
CMD=$CMD" opt.generator.hidden_weight_initializer=kaiming_uniform opt.generator.hidden_bias_initializer=normal"
CMD=$CMD" opt.generator.out_weight_initializer=xavier_uniform opt.generator.out_bias_initializer=normal"
CMD=$CMD" opt.discriminator.hidden_weight_initializer=kaiming_uniform opt.discriminator.hidden_bias_initializer=normal"
CMD=$CMD" opt.discriminator.out_weight_initializer=xavier_uniform opt.discriminator.out_bias_initializer=normal"

for m in $(seq 0 $((ensembleSize-1)));
do

  a=$((m * nNodes + 1))
  b=$((a + step))
  idx=$((m + offset))

  cat ${PBS_NODEFILE} | sed -n "${a},${b}p" > newhost
  cat newhost

  resultLoc=$resultLocBase"_v"$idx
  echo "submit job "$idx
  echo "driver: "$driver
  mpiexec -n $nRanks -ppn 4 -hostfile newhost python $driver hydra.run.dir=$resultLoc $CMD &
  python -c "print('something is running here')"
  echo "job submitted"
  sleep 1
  rm newhost

done

wait
