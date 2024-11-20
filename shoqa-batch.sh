REGION=us-west-2
CONFIG_NAME=1119-dorm-mix-4node-p4de
export GPFS="/dorm-workspace"

declare -A queues
queues[FS-P4DE_24XL-Training-us-west-2a]=p4de.24xlarge
#queues[FS-P5_48XL-Training-us-west-2b]=p5.48xlarge
#queues[FS-P5_48XL-Training-us-west-2b-65k-IPv4]=p5.48xlarge


for queue in "${!queues[@]}"; do
    JOB_QUEUE=$queue
    INSTANCE_TYPE=${queues[$queue]}

    JOB_PRIORITY=9999
    SHARE_IDENTIFIER=NeoTeam # NeoTeam, Smaller4NodeTraining, Evaluation
#    MLFLOW_TRACKING_URI=https://prod.$REGION.internal.mlflow.nile.amazon.dev

    NUM_NODES=4
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
    RUNCMD="mkdir -p ${GPFS} && \
            ls -l && cd ${GPFS} && \
            if [ \"\${AWS_BATCH_JOB_NODE_INDEX:-0}\" = \"0\" ]; then \
                echo 'Node 0: Downloading code from S3...' && \
                aws s3 sync s3://shopqa-users/ronzhi/Mistral-NeMo-12B-Instruct ./Mistral-NeMo-12B-Instruct && \
                aws s3 sync s3://shopqa-users/ronzhi/12b ./12b && \
                touch download_complete; \
            else \
                echo 'Non-main node waiting for downloads to complete...' && \
                while [ ! -f download_complete ]; do sleep 10; done; \
            fi && \
            cd 12b/NeMo-Aligner && \
            chmod +x *.sh && \
            ${DDP_ENVS} ./shopqa-dorm-mix-train.sh"

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
                    "sourcePath": "/tmp"
                },
                "name": "tmp-volume"
            },
            {
                "host": {
                    "sourcePath": "/var/tmp"
                },
                "name": "var-tmp-volume"
            }
        ],
        "mountPoints": [
            {
                "sourceVolume": "tmp-volume",
                "containerPath": "/tmp",
                "readOnly": false
            },
            {
                "sourceVolume": "var-tmp-volume",
                "containerPath": "/var/tmp",
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