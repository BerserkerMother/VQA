CUDA_VISABLE_DEVICES=0 python3 src/train.py --data data/ --batch_size 128 --glove data/glove --candidate_ans data/answer_dict.json --ans_freq 8 --qu_max 14 --num_workers 4 --d_model 256 --attention_dim 256 --num_heads 4 --num_layers 6 --dropout .2 --lr 1e-4 --momentum .9 --weight_decay 0 --epochs 30 --ex_name 'one bert encoder(baseline) using sigmoid and treat problem like binary cross entropy' 