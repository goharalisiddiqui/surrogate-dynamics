#!/bin/sh

####### PREPARE ENV #######
if [ ! -d ".venv" ]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi
########################################

unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs

else 
    echo "Running on local machine"
    pref=''
fi

script="trainer.py"
extra=""
#Read command line arguments
while getopts dm:c:s: flag
do
    case "${flag}" in
        d) extra="${extra} --debug";;
        m) script="${OPTARG}.py";;
        c) config="${OPTARG}";;
        s) script="${OPTARG}";;
    esac
done


$pref python ${script} --config ${config} ${extra}