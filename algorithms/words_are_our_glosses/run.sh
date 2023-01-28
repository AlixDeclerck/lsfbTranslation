#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/train.fr --train-tgt=./data/train.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "train_small" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/train.fr --train-tgt=./data/train.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json --cuda --embed-size=128 --hidden-size=128
elif [ "$1" = "train_small_data" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/train_small.fr --train-tgt=./data/train_small.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json --cuda --embed-size=128 --hidden-size=128
elif [ "$1" = "test" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./data/test.fr ./data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python3 run.py train --train-src=./data/train.fr --train-tgt=./data/train.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json
elif [ "$1" = "train_local_small" ]; then
	python3 run.py train --train-src=./data/train.fr --train-tgt=./data/train.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json --embed-size=128 --hidden-size=128
elif [ "$1" = "train_local_small_data" ]; then
	python3 run.py train --train-src=./data/train_small.fr --train-tgt=./data/train_small.en --dev-src=./data/dev.fr --dev-tgt=./data/dev.en --vocab=vocab.json --embed-size=128 --hidden-size=128
elif [ "$1" = "test_local" ]; then
    python3 run.py decode model.bin ./data/test.fr ./data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./data/train.fr --train-tgt=./data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
