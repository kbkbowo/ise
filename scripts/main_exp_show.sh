python main_exp_show_rn.py \
-t push_bar \
-n results_main/baseline_dino_naive_push_bar.json \
-rj results_main/baseline_dino_rej_push_bar.json \
-rt results_main/baseline_dino_retrive_push_bar.json \
-a results_main/all_push_bar.json \
-tt 16 \
-r

python main_exp_show_rn.py \
-t pick_bar \
-n results_main/baseline_dino_naive_pick_bar.json \
-rj results_main/baseline_dino_rej_pick_bar.json \
-rt results_main/baseline_dino_retrive_pick_bar.json \
-a results_main/all_pick_bar.json \
-tt 2 \
-r

python main_exp_show_rn.py \
-t open_box \
-n results_main/baseline_dino_naive_open_box.json \
-rj results_main/baseline_dino_rej_open_box.json \
-rt results_main/baseline_dino_retrive_refine_scratch_open_box.json \
-a results_main/all_open_box.json \
-tt 2 \
-r

python main_exp_show_rn.py \
-t turn_faucet \
-n results_main/baseline_dino_naive_turn_faucet.json \
-rj results_main/baseline_dino_rej_turn_faucet.json \
-rt results_main/baseline_dino_retrive_refine_scratch_turn_faucet.json \
-a results_main/all_turn_faucet.json \
-tt 2 \
-r

# python main_exp_show_sr.py \
# -t push_bar \
# -n results_main/baseline_dino_naive_push_bar.json \
# -rj results_main/baseline_dino_rej_push_bar.json \
# -rt results_main/baseline_dino_retrive_refine_scratch_push_bar.json \
# -a results_main/all_push_bar.json \
# -tt 16

# python main_exp_show_sr.py \
# -t pick_bar \
# -n results_main/baseline_dino_naive_pick_bar.json \
# -rj results_main/baseline_dino_rej_pick_bar.json \
# -rt results_main/baseline_dino_retrive_refine_scratch_pick_bar.json \
# -a results_main/all_pick_bar.json \
# -tt 2

# python main_exp_show_sr.py \
# -t open_box \
# -n results_main/baseline_dino_naive_open_box.json \
# -rj results_main/baseline_dino_rej_open_box.json \
# -rt results_main/baseline_dino_retrive_refine_scratch_open_box.json \
# -a results_main/all_open_box.json \
# -tt 2

# python main_exp_show_sr.py \
# -t turn_faucet \
# -n results_main/baseline_dino_naive_turn_faucet.json \
# -rj results_main/baseline_dino_rej_turn_faucet.json \
# -rt results_main/baseline_dino_retrive_refine_scratch_turn_faucet.json \
# -a results_main/all_turn_faucet.json \
# -tt 2