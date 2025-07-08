python code/run_model_inference.py \
    --input_path output/stage2_rot_fixing/Etiquette_gpt4_3opt_final_data.csv \
    --output_path output/ \
    --model 'llama2-7b-chat' \
    --conditioning cval \
    --temperature_list 1e-3 \
    --type_of_setup vllm 