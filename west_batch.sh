REGION=us-west-2
CONFIG_NAME=1113_12b_rm_west2a_oh_100k_2epoch_2node_lr3e6
export GPFS="/fsx-Training/home/ronzhi/12b/NeMo-Aligner"

declare -A queues
queues[FS-P4DE_24XL-Training-us-west-2a]=p4de.24xlarge


for queue in "${!queues[@]}"; do
    JOB_QUEUE=$queue
    INSTANCE_TYPE=${queues[$queue]}

    JOB_PRIORITY=100
    SHARE_IDENTIFIER=Smaller4NodeTraining # NeoTeam, Smaller4NodeTraining, Evaluation
#    MLFLOW_TRACKING_URI=https://prod.$REGION.internal.mlflow.nile.amazon.dev

    NUM_NODES=2
    NUM_GPUS_PER_NODE=8
    NUM_CPUS_PER_NODE=96
    MEMORY_PER_NODE=1143265
    WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $NUM_NODES))
    ECR_DOCKER_IMAGE=684288478426.dkr.ecr.us-east-1.amazonaws.com/nemo:aligner-nemo-24.07-mistral # CHANGE HERE

    # Run docker image on AWS Batch
    echo "Registering job definition"
    random_job_suffix=$(printf '%s' $(openssl rand -hex 12) | cut -c 1-10)
    JOB_NAME=$(echo "${CONFIG_NAME}" | sed 's/\./_/g')-${SHARE_IDENTIFIER}-${random_job_suffix}
    JOB_DEFINITION_NAME=nemo-aligner-${random_tag}-${JOB_NAME}
    JOB_TYPE=multinode
    PATH_ENVS='PYTHONPATH=$GPFS:$PYTHONPATH'

    # if NUM_NODES > 1, then use DDP
    if [ $NUM_NODES -gt 1 ]; then
        # FYI: NCCL_DEBUG=INFO is used for nccl debugging
        DDP_ENVS='MASTER_ADDR=${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:=$(hostname -f)} NCCL_SOCKET_IFNAME=eth0 MASTER_PORT=1234 WORLD_SIZE='${WORLD_SIZE}' NODE_RANK=${AWS_BATCH_JOB_NODE_INDEX}'
    else
        DDP_ENVS=''
    fi
    RUNCMD="ls -l && ls -l ${GPFS} && cd ${GPFS} && ${DDP_ENVS} ./mixdata_rm_train_batch_west.sh"
#    RUN_ARGS='run.user='${USER}'-nemo-aws-batch  run.name=aws-batch-'${JOB_NAME}
#   RUNCMD_LIST=("sh" "-c" "${DDP_ENVS} ${RUNCMD} ${RUN_ARGS}")

    PIP_INSTALL="cd ${GPFS}; pip install -e .; pip install mlflow"
    RUNCMD_LIST=("sh" "-c" "${RUNCMD}")

    # RUNCMD_LIST=("sh" "-c" "${PATH_ENVS} ${RUNCMD}")
    # RUNCMD_LIST=("sh" "-c" "${ROTARY_EMBEDDING_SWAP} ${DDP_ENVS} ${PATH_ENVS} ${RUNCMD}")
    RUNCMD_LIST_STR=$(printf '%s\n' "${RUNCMD_LIST[@]}" | jq -R . | jq -s .)
    echo "DOCKER RUN CMD: $RUNCMD_LIST_STR"

    CONTAINER_PROPERTIES='{
        "image": "'$ECR_DOCKER_IMAGE'",
        "command": '$RUNCMD_LIST_STR',
        "privileged": true,
        "instanceType": "'$INSTANCE_TYPE'",
        "resourceRequirements": [
            {
                "value": "'${NUM_GPUS_PER_NODE}'",
                "type": "GPU"
            },
            {
                "value": "'${NUM_CPUS_PER_NODE}'",
                "type": "VCPU"
            },
            {
                "value": "'${MEMORY_PER_NODE}'",
                "type": "MEMORY"
            }
        ],
        "ulimits": [
            {
                "name": "memlock",
                "softLimit": -1,
                "hardLimit": -1
            },
            {
                "name": "stack",
                "softLimit": 67108864,
                "hardLimit": 67108864
            }
        ],
        "volumes": [
        {
          "host": {
            "sourcePath": "/fsx-Training"
          },
          "name": "local-fsx-volume"
        },
        {
            "host": {
                "sourcePath": "/fsx-Training/home/$USER/12b/NeMo-Aligner/tmp"
            },
            "name": "tmp-volume"
        },
        {
            "host": {
                "sourcePath": "/fsx-Training/home/$USER/12b/NeMo-Aligner/var/tmp"
            },
            "name": "var-tmp-volume"
        },
        {
            "host": {
                "sourcePath": "/fsx-Training/home/$USER/12b/NeMo-Aligner/log"
            },
            "name": "log-volume"
        },
        {
            "host": {
                "sourcePath": "/fsx-Training/home/$USER/12b/NeMo-Aligner/results_model/"
            },
            "name": "results-model-volume"
        }
        ],
        "mountPoints": [
            {
              "sourceVolume": "local-fsx-volume",
              "containerPath": "/fsx-Training",
              "readOnly": false
            },
            {
                "sourceVolume": "tmp-volume",
                "containerPath": "/tmp",
                "readOnly": false
            },
            {
                "sourceVolume": "var-tmp-volume",
                "containerPath": "/var/tmp",
                "readOnly": false
            },
            {
                "sourceVolume": "log-volume",
                "containerPath": "/var/log",
                "readOnly": false
            },
            {
                "sourceVolume": "results-model-volume",
                "containerPath": "/results",
                "readOnly": false
            }
        ],
        "linuxParameters": {
            "devices": [
                {
                "hostPath": "/dev/infiniband/uverbs0",
                "containerPath": "/dev/infiniband/uverbs0",
                "permissions": [
                    "READ",
                    "WRITE",
                    "MKNOD"
                ]
                },
                {
                "hostPath": "/dev/infiniband/uverbs1",
                "containerPath": "/dev/infiniband/uverbs1",
                "permissions": [
                    "READ",
                    "WRITE",
                    "MKNOD"
                ]
                },
                {
                "hostPath": "/dev/infiniband/uverbs2",
                "containerPath": "/dev/infiniband/uverbs2",
                "permissions": [
                    "READ",
                    "WRITE",
                    "MKNOD"
                ]
                },
                {
                "hostPath": "/dev/infiniband/uverbs3",
                "containerPath": "/dev/infiniband/uverbs3",
                "permissions": [
                    "READ",
                    "WRITE",
                    "MKNOD"
                ]
                }
            ],
            "sharedMemorySize": 1143265,
            "tmpfs": []
        },
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {},
            "secretOptions": []
        }
    }'
    NODE_PROP_STR='{
            "numNodes":'${NUM_NODES}',
            "mainNode":0,
            "nodeRangeProperties": [{
                "targetNodes": "0:",
                "container": '$CONTAINER_PROPERTIES'
            }]
        }'
    NODE_PROP_STR=$(echo "$NODE_PROP_STR" | tr -d '\n')
    response_reg_job_def=$(aws batch register-job-definition \
        --job-definition-name $JOB_DEFINITION_NAME \
        --type $JOB_TYPE \
        --scheduling-priority $JOB_PRIORITY \
        --node-properties "$NODE_PROP_STR" \
        --region $REGION \
        --output table)
    echo "response_reg_job_def: $response_reg_job_def"
    echo "Registered job definition $JOB_DEFINITION_NAME"

    # waiting job definition register become active.
    sleep 10

    echo "Submitting job"
    aws batch submit-job --job-name $JOB_NAME --job-queue $JOB_QUEUE  --job-definition $JOB_DEFINITION_NAME --share-identifier $SHARE_IDENTIFIER --region $REGION
    echo "Submitted job $JOB_NAME"
done