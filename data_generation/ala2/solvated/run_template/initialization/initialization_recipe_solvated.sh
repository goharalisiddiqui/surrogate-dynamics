#!/bin/bash
#SBATCH -J MD_init
### number of cores
#SBATCH -c 4
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -p zencloud
#SBATCH --mem=8G
#SBATCH --time=100:00:00
#SBATCH --export=ALL
#SBATCH -o ./slurm_logs/slurm-%J.out
#SBATCH -e ./slurm_logs/slurm-%J.err

####### PREPARE ENV #######
echo "Loading modules from $SLURM_PRESCRIPT_GROMACS :"
cat $SLURM_PRESCRIPT_GROMACS
source $SLURM_PRESCRIPT_GROMACS
########################################

####### SETTING ENV VARIABLE #######
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
####################################

unset $pref
if [ ! -z "${SLURM_JOB_ID}" ]; then
    echo "Running on compute node"
    pref='srun'
    mkdir -p slurm_logs
else 
    echo "Running on local machine"
    pref=''
fi

#############################################
# Environment Initilization
#############################################
# Clean directory of all files except the recipec
ls --hide=initialization_recipe*.sh --hide=slurm_setup.sh --hide=input_files | xargs -d '\n' rm
# GROMACS RECIPE FOR INITIALIZING MD SIMULATIONS
set -e
#############################################
# Vacuum Initilization
#############################################
echo "VACUUM INITIALIZATION:"
 
echo "Step 01 - choose Force field and create topology"
echo -e "1\n1" | gmx_mpi pdb2gmx -f input_files/initial_structure.pdb -ignh -o conf_initial.gro -p topol_initial.top -i posre_initial.itp &> out_1-1.txt

echo "Step 02 - Define Simulation Box"
gmx_mpi editconf -f conf_initial.gro -o conf_initial_boxed.gro -c -d 1.0 -bt cubic &> out_1-2.txt

echo "Step 03 - Prepare for vacuum minimization"
gmx_mpi grompp -f input_files/minim.mdp -c conf_initial_boxed.gro -p topol_initial.top -po mdp_initial.mdp -o tpr_initial.tpr &> out_1-3.txt

echo "Step 04 - Run minimization in vacuum"
$pref gmx_mpi mdrun -s tpr_initial.tpr -deffnm vm &> out_1-4.txt

echo "Optional Step 05 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f vm.trr -s tpr_initial.tpr -o vm.pdb -pbc mol -center &> out_1-5.txt 

echo -e "DONE\n\n"

#############################################
# Solvation and Energy Minimization
#############################################
echo "SOLVATION AND MINIMIZATION:"

echo "Step 01 - Add Water molecules"
cp topol_initial.top topol_solvated.top # To keep all the topology files for records
gmx_mpi solvate -cp vm.gro -p topol_solvated.top -cs spc216.gro -o conf_solvated.gro &> out_2-1.txt

echo "Step 02 - Neutralize with NACL ions"
cp topol_solvated.top topol_solvated_neutral.top # To keep all the topology files for records
gmx_mpi grompp -maxwarn 1 -f input_files/em.mdp -c conf_solvated.gro -p topol_solvated_neutral.top -po em_charged.mdp -o em_charged.tpr &> out_2-2.txt # gmx genion require .tpr files to neutralize the system this we make one
echo "SOL" | gmx_mpi genion -neutral -pname NA -nname CL -s em_charged.tpr -o conf_solvated_neutral.gro -p topol_solvated_neutral.top &>> out_2-2.txt

echo "Step 03 - Prepare for minimization" 
gmx_mpi grompp -f input_files/em.mdp -c conf_solvated_neutral.gro -p topol_solvated_neutral.top -po em.mdp -o em.tpr &> out_2-3.txt
  
echo "Step 04 - Run minimization in water"
$pref gmx_mpi mdrun -s em.tpr -deffnm em &> out_2-4.txt

echo "Optional Step 05 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f em.trr -s em.tpr -o em.pdb -pbc mol -center &> out_2-5.txt

echo -e "DONE\n\n"

#############################################
# NVT Equilibration
#############################################
echo "NVT EQUILIBRATION:"

echo "Step 01 - PreProcess"
gmx_mpi grompp -f input_files/nvt.mdp -c em.gro -r em.gro -p topol_solvated_neutral.top -po nvt.mdp -o nvt.tpr &> out_3-1.text

echo "Step 02 - Equilibration"
$pref gmx_mpi mdrun -s nvt.tpr -deffnm nvt &> out_3-2.text

echo "Optional Step 03 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f nvt.xtc -s nvt.tpr -o nvt.pdb -pbc mol -center &> out_3-3.txt

echo -e "DONE\n\n"

#############################################
# NPT Equilibration
#############################################

echo "NPT EQUILIBRATION:"

echo "Step 01 - PreProcess"
gmx_mpi grompp -f input_files/npt.mdp -c nvt.gro -r nvt.gro -p topol_solvated_neutral.top -po npt.mdp -o npt.tpr &> out_4-1.text

echo "Step 02 - Equilibration"
$pref gmx_mpi mdrun -s npt.tpr -deffnm npt &> out_4-2.text

echo "Optional Step 03 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f npt.xtc -s npt.tpr -o npt.pdb -pbc mol -center &> out_4-3.txt

echo -e "DONE\n\n"

#############################################
# Prepare for MD
#############################################

echo "PREPROCESSING MD...."
gmx_mpi grompp -f input_files/md.mdp -c npt.gro -p topol_solvated_neutral.top -po md.mdp -o md.tpr &> out_5-1.text
echo -e "ALL DONE\n\n"
rm \#*