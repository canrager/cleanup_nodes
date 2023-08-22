echo "Running script for fig1"
python fig1_resid_onto_head_lineplots.py

echo "Running script for fig2"
python fig2_cleanup_barplots.py

# echo "Running script for fig3"
# python fig3_patch_v_input_cleaners.py

echo "Running script for fig4"
python fig4_DLA_resample_ablation.py

echo "Running script for fig5"
python fig5_DLA_scatter_plot.py
