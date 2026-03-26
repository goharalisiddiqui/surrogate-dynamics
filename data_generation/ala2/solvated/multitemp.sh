for temp in 200
do
    cp -r template ${temp}K

    cd ${temp}K/initialization/
    sed -i "s/.*ref_t.*/ref_t \t\t\t\t\t= ${temp} \t\t ; reference temperature/g" input_files/nvt.mdp
    sed -i "s/.*ref_t.*/ref_t \t\t\t\t\t= ${temp} \t\t ; reference temperature/g" input_files/npt.mdp
    sed -i "s/.*ref_t.*/ref_t \t\t\t\t\t= ${temp} \t\t ; reference temperature/g" input_files/md.mdp
    bash initialization_recipe_solvated.sh
    cd ../ala2_100ns/
    sbatch run_script.sh
    cd ../..
done