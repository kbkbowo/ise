python main_exp.py -s 400 --n 4 --retrieve_on --refine_step 0 \
--out_file "results_new/new_pick_bar_all.json" \
--out_dir "results_new/new_pick_bar_all" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name "pick_bar"