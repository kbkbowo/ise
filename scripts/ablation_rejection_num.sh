num=$1
task=$2

python main_exp.py -s 400 --n $num\
--out_file "ablation_rejection_num/$num_$task.json" \
--out_dir "ablation_rejection_num/$num_$task" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name "pick_bar"