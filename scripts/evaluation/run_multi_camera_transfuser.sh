TRAIN_TEST_SPLIT=navtest
CHECKPOINT=$1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=multi_camera_transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=multi_camera_transfuser_agent_eval 
