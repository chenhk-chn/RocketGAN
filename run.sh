CUDA_VISIBLE_DEVICES=3 python main.py --log_dir ./logs/rocketgan_market2duke_all --tag v1.2 \
-e 0 1 2 3 4 5 6 7 \
--source_dataset market --target_dataset duke

CUDA_VISIBLE_DEVICES=3 python main.py --log_dir ./logs/rocketgan_market2duke_sep --tag v1.2 \
-e 0 1 2 3 \
--source_dataset market --target_dataset duke

CUDA_VISIBLE_DEVICES=3 python main.py --log_dir ./logs/rocketgan_market2duke_sep --tag v1.2 \
-e 4 5 6 7 \
--source_dataset market --target_dataset duke