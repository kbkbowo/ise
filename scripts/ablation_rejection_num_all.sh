num=$1
task=$2

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all_scratch.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all_scratch" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_from_scratch \
--refine_step 100

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all_scratch.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all_scratch" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_from_scratch \
--refine_step 100

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all_scratch.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all_scratch" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_from_scratch \
--refine_step 100

python main_exp.py -s 400 --n $num \
--out_file "ablation_rejection_num/${num}_${task}_all_scratch.json" \
--out_dir "ablation_rejection_num/${num}_${task}_all_scratch" \
--temperature 0.3 \
--agg_metric "min" \
--enc_method "dinov2" \
--task_name ${task} \
--retrieve_on \
--refine_from_scratch \
--refine_step 100