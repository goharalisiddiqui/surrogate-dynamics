####### PREPARE ENV #######
if [ ! -z "${SLURM_PRESCRIPT_ML}" ]; then
    echo "Loading modules from $SLURM_PRESCRIPT_ML"
    source $SLURM_PRESCRIPT_ML
else
    if [ ! -d ".venv" ]; then
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi
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

#Read command line arguments
while getopts d flag
do
    case "${flag}" in
        d) debug=1;;
    esac
done
