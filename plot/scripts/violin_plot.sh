python plot/violin_plot.py \
-f1 ablation_rejection_num/naive_open_box.json \
ablation_rejection_num/2_open_box.json \
ablation_rejection_num/retrieve_open_box.json \
ablation_rejection_num/2_open_box_all.json \
-c1 naive rejection retrieval all \
-n1 Task_Open_Box \
-f2 ablation_rejection_num/2_pick_bar.json \
ablation_rejection_num/2_pick_bar_all.json \
-c2 rejection all \
-n2 Task_Pick_Bar \
-f3 ablation_rejection_num/naive_push_bar.json \
ablation_rejection_num/2_push_bar.json \
ablation_rejection_num/retrieve_push_bar.json \
ablation_rejection_num/2_push_bar_all.json \
-c3 naive rejection retrieval all \
-n3 Task_Push_Bar \
-f4 ablation_rejection_num/naive_turn_faucet.json \
ablation_rejection_num/2_turn_faucet.json \
ablation_rejection_num/retrieve_turn_faucet.json \
ablation_rejection_num/2_turn_faucet_all.json \
-c4 naive rejection retrieval all \
-n4 Task_Turn_Faucet \