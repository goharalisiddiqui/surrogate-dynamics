#! /bin/bash
#############################################
# Environment Initilization
#############################################
# Clean directory of all files except the recipec
ls --hide=initialization_recipe*.sh --hide=slurm_setup.sh --hide=input_files | xargs -d '\n' rm
# GROMACS RECIPE FOR INITIALIZING MD SIMULATIONS
set -e
# Activate module
module load gromacs/2024.3
#############################################
# Vacuum Initilization
#############################################
echo "VACUUM INITIALIZATION:"
 
echo "Step 01 - choose Force field and create topology"
echo -e "1\n7" | gmx_mpi pdb2gmx -f input_files/initial_structure.pdb -ignh -o conf_initial.gro -p topol_initial.top -i posre_initial.itp &> out_1-1.txt

echo "Step 02 - Define Simulation Box"
gmx_mpi editconf -f conf_initial.gro -o conf_initial_boxed.gro -c -d 1.0 -bt cubic &> out_1-2.txt

echo "Step 03 - Prepare for vacuum minimization"
gmx_mpi grompp -f input_files/minim.mdp -c conf_initial_boxed.gro -p topol_initial.top -po minim_out.mdp -o tpr_initial.tpr &> out_1-3.txt

echo "Step 04 - Run minimization in vacuum"
gmx_mpi mdrun -s tpr_initial.tpr -deffnm vm &> out_1-4.txt

echo "Optional Step 05 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f vm.trr -s tpr_initial.tpr -o vm.pdb -pbc mol -center &> out_1-5.txt 

echo -e "DONE\n\n"

#############################################
# NVT Equilibration
#############################################
echo "NVT EQUILIBRATION:"


echo "Step 01 - PreProcess"
gmx_mpi grompp -f input_files/nvt.mdp -c vm.gro -r vm.gro -p topol_initial.top -po nvt_out.mdp -o nvt.tpr &> out_2-1.text

echo "Step 02 - Equilibration"
gmx_mpi mdrun -s nvt.tpr -deffnm nvt &> out_2-2.text

echo "Optional Step 03 - Convert to pdb for viewing"
echo -e "1\n0" | gmx_mpi trjconv -f nvt.xtc -s nvt.tpr -o nvt.pdb -pbc mol -center &> out_2-3.txt

echo -e "DONE\n\n"

#############################################
# Prepare for MD
#############################################

echo "PREPROCESSING MD...."
gmx_mpi grompp -f input_files/md.mdp -c nvt.gro -p topol_initial.top -po md_mod.mdp -o md.tpr &> out_5-1.text
echo -e "ALL DONE\n\n"
rm \#*