GPU_ID=2


# SciERC,  --use_ner_results: use the original entity type predicted by NER models
DATE_DATA=2025-05-14
for seed in 42 ; do 
NER_MODEL_PATH=gsap/models/$DATE_DATA/PL-Marker-gsap-scibert-$seed
echo $NER_MODEL_PATH
#--use_ner_results \
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased --do_lower_case  \
	--project_name plmarker-rel-gsap \
    --run_name hund \
    --learning_rate 2e-5  --num_train_epochs 6  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16  --seed $seed \
    --data_dir $NER_MODEL_PATH \
    --train_file ent_pred_train.json \
    --dev_file ent_pred_dev.json  \
    --test_file ent_pred_test.json  \
    --output_dir gsap/models/$DATE_DATA/rel-scibert-$seed  --overwrite_output_dir
done;
