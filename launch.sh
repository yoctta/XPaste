#python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
#python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html	
# pip install -r requirements.txt
#pip install -U timm
export DETECTRON2_DATASETS=/mnt/data/detectron2_datasets
#python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=29500 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

GPUS=`nvidia-smi -L | wc -l`

echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

export OMP_NUM_THREADS=4
# export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

# python3 -m torch.distributed.launch --nnodes ${NODE_COUNT} \
#         --node_rank ${RANK} \
#         --master_addr ${MASTER_ADDR} \
#         --master_port ${MASTER_PORT} \
#         --nproc_per_node ${GPUS} \
# 	train_net.py \
#   --num-machines 2 \
# 	"$@"

python train_net.py  \
        --machine-rank ${RANK} \
        --num-machines ${NODE_COUNT}\
        --num-gpus ${GPUS} \
	"$@"
