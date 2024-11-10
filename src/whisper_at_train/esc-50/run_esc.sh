#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/whisper-a/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models


CosineAnnealingLR=False
CosTmax=15
Cosmin=1e-5
lr=1e-4
freqm=0
timem=10
mixup=0.5
batch_size=48

model=whisper-high-wa_ctr_2_5_10_20 #whisper-high-lw_tr_1_8 (tl-tr, lr=5e-5) whisper-high-lw_down_tr_512_1_8 (tl-tr-512, w/ low-dim proj, lr=1e-4)
cuda=0 #用哪个显卡，0是第一个，1是第二个

model_size=large-v1 #
tdim=100

dataset=esc
bal=bal
epoch=30

lrscheduler_start=7     # 这三行学习率的超参，很重要，lrscheduler_start是哪个epoch开始衰减
lrscheduler_decay=0.9   #lrscheduler_decay衰减率，lrscheduler_step 几个epoch衰减一次
lrscheduler_step=1

wa=False
wa_start=16
wa_end=40
lr_adapt=False
warmup=True


label_smooth=0

base_exp_dir=./exp/test-${dataset}-${model}-cos${CosineAnnealingLR}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-bs${batch_size}-lda${lr_adapt}-mix${mixup}-${freqm}-${timem}-tdim=${tdim}

folders=("fold1" "fold2" "fold3" "fold4" "fold5")

# 检查 result.csv 是否在所有 fold 文件夹中都存在
all_result_exist=true
start_fold=1

for folder in "${folders[@]}"; do
  if [ ! -d "$base_exp_dir/$folder" ]; then
    all_result_exist=false
    start_fold=${folder:4}
    break
  fi
  
  if [ ! -f "$base_exp_dir/$folder/stats_${epoch}.pickle" ]; then
    all_result_exist=false
    start_fold=${folder:4}
    break
  fi
done


if $all_result_exist ; then
  echo 'All results exist'
  exit
else
  echo "Results missing, starting from fold${start_fold}"
fi

for((fold=start_fold;fold<=5;fold++));
do
  echo "now process fold${fold}"
  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=/vasp/exp/whisper/data/datafiles/esc_train_data_${fold}.json
  te_data=/vasp/exp/whisper/data/datafiles/esc_eval_data_${fold}.json
  label_csv=/vasp/exp/whisper/data/datafiles/esc_class_labels_indices.csv

  train_tar_path=/vasp/exp/whisper/data/yourfeaturepath
  eval_tar_path=/vasp/exp/whisper/data/yourfeaturepath

  CUDA_VISIBLE_DEVICES=${cuda} python -W ignore ../runs.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label_csv ${label_csv} --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --model_size ${model_size} --label_smooth ${label_smooth} \
  --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
  --loss CE --metrics acc --warmup True \
  --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
  --num-workers 8 --CosineAnnealingLR ${CosineAnnealingLR} --CosTmax ${CosTmax} --Cosmin ${Cosmin} --tdim ${tdim} \
  --train_tar_path ${train_tar_path} --eval_tar_path ${eval_tar_path}
done

python ./get_result.py --exp_path ${base_exp_dir}