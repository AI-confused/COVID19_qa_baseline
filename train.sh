export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_qa.py \
--model_name_or_path ../chinese_wwm_ext_pytorch \
--do_train \
--do_eval \
--es_index passages \
--es_ip localhost \
--data_dir ./data/data_0 \
--test_dir ./data/test1.csv \
--passage_dir ./data/passage1.csv \
--output_dir ./output/ \
--max_seq_length 512 \
--max_question_length 96 \
--eval_steps 50 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 1e-5 \
--train_steps 1000