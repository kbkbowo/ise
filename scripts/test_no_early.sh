EXP_PATH="/home/seanfu/ise/results_no_early_stop"

TASK1="push_bar"
TASK2="pick_bar"
TASK3="open_box"
TASK4="turn_faucet"

python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK1.json --out_dir $EXP_PATH/$TASK1 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK1
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK1.json --out_dir $EXP_PATH/$TASK1 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK1
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK1.json --out_dir $EXP_PATH/$TASK1 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK1
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK1.json --out_dir $EXP_PATH/$TASK1 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK1

python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK2.json --out_dir $EXP_PATH/$TASK2 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK2
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK2.json --out_dir $EXP_PATH/$TASK2 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK2
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK2.json --out_dir $EXP_PATH/$TASK2 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK2
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK2.json --out_dir $EXP_PATH/$TASK2 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK2

python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK3.json --out_dir $EXP_PATH/$TASK3 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK3
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK3.json --out_dir $EXP_PATH/$TASK3 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK3
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK3.json --out_dir $EXP_PATH/$TASK3 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK3
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK3.json --out_dir $EXP_PATH/$TASK3 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK3

python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK4.json --out_dir $EXP_PATH/$TASK4 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK4
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK4.json --out_dir $EXP_PATH/$TASK4 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK4
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK4.json --out_dir $EXP_PATH/$TASK4 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK4
python main_no_early.py --refine_from_scratch --seed 50 --n 4 --retrieve_on --out_file $EXP_PATH/$TASK4.json --out_dir $EXP_PATH/$TASK4 --dist_metric "l2" --temperature 0.3 --agg_metric "min" --enc_method "dinov2" --task_name $TASK4




