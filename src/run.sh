# 0.466 OOF, 0.462 LB
python train.py --fold 0 --model microsoft/deberta-large --batch_size 16 --max_len 256 --learning_rate 1 --epochs 200 --num_samples 5000 --early_stopping_patience 5 2>&1 | tee logs0.txt
python train.py --fold 1 --model microsoft/deberta-large --batch_size 16 --max_len 256 --learning_rate 1 --epochs 200 --num_samples 5000 --early_stopping_patience 5 2>&1 | tee logs1.txt
python train.py --fold 2 --model microsoft/deberta-large --batch_size 16 --max_len 256 --learning_rate 1 --epochs 200 --num_samples 5000 --early_stopping_patience 5 2>&1 | tee logs2.txt
python train.py --fold 3 --model microsoft/deberta-large --batch_size 16 --max_len 256 --learning_rate 1 --epochs 200 --num_samples 5000 --early_stopping_patience 5 2>&1 | tee logs3.txt
python train.py --fold 4 --model microsoft/deberta-large --batch_size 16 --max_len 256 --learning_rate 1 --epochs 200 --num_samples 5000 --early_stopping_patience 5 2>&1 | tee logs4.txt
