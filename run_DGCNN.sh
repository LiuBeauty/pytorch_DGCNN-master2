#!/bin/bash

# input arguments
DATA="${1-MUTAG}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-1} # which fold as testing data
test_number=171 # ifa specified, use the last test_number graphs as test data

# general settings
gm=DGCNN  # model
gpu_or_cpu=gpu
GPU=0  # select the GPU number
CONV_SIZE="26-24-24-1"
sortpooling_k=0.5 # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data,final fully-connected layer
n_hidden=128 # final dense layer's hidden size
bsize=1  # batch sizea
dropout=false


# dataset-specific settings
case ${DATA} in
BRCA_NODE300)
  num_epochs=200
  learning_rate=0.00001
  ;;
KIPAN_NODE300)
  num_epochs=200
  learning_rate=0.00001
  ;;
LUNG_NODE300)
  num_epochs=200
  learning_rate=0.00001
  ;;
**)
  num_epochs=200
  learning_rate=0.00001
  ;;
esac


CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    -seed 1 \
    -data $DATA \
    -fold $fold \
    -learning_rate $learning_rate \
    -num_epochs $num_epochs \
    -hidden $n_hidden \
    -latent_dim $CONV_SIZE \
    -sortpooling_k $sortpooling_k \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    -gm $gm \
    -mode $gpu_or_cpu \
    -dropout $dropout \
    -test_number ${test_number}
fi
