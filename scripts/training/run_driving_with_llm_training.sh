python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=driving_with_llm \
experiment_name=training_driving_with_llm \
train_test_split=navtrain \
dataloader.params.batch_size=1 \
cache_path=/root/workdir/NAVSIM/exp/training_cache_driving_with_llm \
trainer.params.strategy=ddp_find_unused_parameters_true
# use_cache_without_dataset=true \
# force_cache_computation=false \
# cache_path=/root/workdir/NAVSIM/exp/training_cache_driving_with_llm \