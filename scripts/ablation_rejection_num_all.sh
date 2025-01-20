num=$1
task=$2

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_step 0

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_step 0

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_step 0

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_step 0