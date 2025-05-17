GPU_ID=1

# For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5

# ACE05
DATE_DATA=2025-05-14
mkdir -p gsap/models/${DATE_DATA}
for lr in 1e-5 2e-5 5e-5; do 
seed=42
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased  --do_lower_case  \
	--project_name plmarker-gsap \
    --run_name gazelle-lr${lr} \
    --data_dir ../data_gsap_plmarker/${DATE_DATA}/ground_truth/clean/SentenceFootnote/  \
    --learning_rate $lr  --num_train_epochs 6  --per_gpu_train_batch_size  16  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 2  \
    --max_seq_length 512  --save_steps 500  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
	--seed $seed  --onedropout  --lminit  \
    --train_file ${DATE_DATA}_train.jsonl --dev_file ${DATE_DATA}_dev.jsonl --test_file ${DATE_DATA}_test.jsonl  \
	--fp16 \
    --output_dir gsap/models/${DATE_DATA}/PL-Marker-gsap-scibert-$seed  --overwrite_output_dir  --output_results
done;
#Average the scores
#python3 sumup.py ace05ner PL-Marker-ace05-bert


