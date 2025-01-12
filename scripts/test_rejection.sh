EXP_PATH="/home/seanfu/ise/results_main/video_l2_guide"

python main_exp.py -g 3.0 --seed 100 --n 4 --retrieve_on --out_file $EXP_PATH.json --out_dir $EXP_PATH --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name "push_bar"
