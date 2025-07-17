TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=modified_transfuser_agent \
use_cache_without_dataset=true \
force_cache_computation=false \
cache_path=/root/workdir/NAVSIM/exp/training_cache \
experiment_name=training_resnet50_transfuser_agent \
train_test_split=$TRAIN_TEST_SPLIT \