#!/bin/bash

data_path="/hdd/Continual_learning_data/FINAL" # path to the data
checkpoint_path="/home/say26747/Desktop/git/BrainCL/OUTPUT" # where to save the checkpoints
load_checkpoint_path_episode="None" # path to the end of an episode checkpoint
load_checkpoint_path_epoch="None" # path to the last epoch checkpoint


# other parameters
batch_size=4
epochs=1
continue_training_from_episode=0 # wherether to load a checkpoint and continue training from it
continue_training_from_epoch=0 # wherether to load a checkpoint and continue training from it
drop_modality=1
num_workers=12
lr=0.001
optimizer="adam"
name="Yousef_MOE"
show_progress=0
seed=-1
compile=0
amp=1
alpha=0.000005
alpha_max=0.6
beta=0.8
dynamic_coef=1
tempurature=2
num_experts=4
context_dim=10
sequence=0 # which sequence to use (S1=0, S2=1)

# run the script
python yousef_MOE.py --data_path $data_path --checkpoint_path $checkpoint_path \
 --batch_size $batch_size --epochs $epochs --compile $compile\
 --continue_training_from_episode $continue_training_from_episode --continue_training_from_epoch $continue_training_from_epoch \
 --load_checkpoint_path_episode $load_checkpoint_path_episode --load_checkpoint_path_epoch $load_checkpoint_path_epoch \
  --drop_modality $drop_modality --num_workers $num_workers --amp $amp \
     --name $name --lr $lr --optimizer $optimizer --show_progress $show_progress --seed $seed \
     --alpha $alpha --beta $beta --num_experts $num_experts --context_dim $context_dim \
     --dynamic_coef $dynamic_coef --tempurature $tempurature --alpha_max $alpha_max \
     --sequence $sequence
