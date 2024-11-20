REGION=us-west-2
CONFIG_NAME=1119-dorm-mix-4node-p5
export GPFS="/dorm-workspace"

declare -A queues
queues[FS-P5_48XL-Training-us-west-2b]=p5.48xlarge
#queues[FS-P4DE_24XL-Training-us-west-2a]=p4de.24xlarge

for queue in "${!queues[@]}"; do
    JOB_QUEUE=$queue
    INSTANCE_TYPE=${queues[$queue]}

    JOB_PRIORITY=9999
    SHARE_IDENTIFIER=Smaller4NodeTraining # NeoTeam, Smaller4NodeTraining, Evaluation

    # P5.48xlarge specifications
    NUM_NODES=4
    NUM_GPUS_PER_NODE=8
    NUM_CPUS_PER_NODE=192  # Changed from 96 for P5
    MEMORY_PER_NODE=1952788  # Changed from 1143265 for P5
    NETWORK_DEVICE_NUM=32   # Changed from 4 for P5
    WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $NUM_NODES))
    ECR_DOCKER_IMAGE=684288478426.dkr.ecr.us-east-1.amazonaws.com/nemo:aligner-nemo-24.07-mistral

    # Generate device properties string for P5's 32 network devices
    DEVICE_PROP_STR=""
    for ((i=0; i<$NETWORK_DEVICE_NUM; i++)); do
        DEVICE_PROP_STR+='{
            "hostPath": "/dev/infiniband/uverbs'$i'",
            "containerPath": "/dev/infiniband/uverbs'$i'",
            "permissions": ["READ", "WRITE", "MKNOD"]
        }'
        if [ $i -lt $(($NETWORK_DEVICE_NUM-1)) ]; then
            DEVICE_PROP_STR+=','
        fi
    done

    # Job definition setup
    random_job_suffix=$(printf '%s' $(openssl rand -hex 12) | cut -c 1-10)
    JOB_NAME=$(echo "${CONFIG_NAME}" | sed 's/\./_/g')-${SHARE_IDENTIFIER}-${random_job_suffix}
    JOB_DEFINITION_NAME=nemo-aligner-p5-${random_job_suffix}-${JOB_NAME}
    JOB_TYPE=multinode
    PATH_ENVS='PYTHONPATH=$GPFS:$PYTHONPATH'

    # DDP configuration
    if [ $NUM_NODES -gt 1 ]; then
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

    RUNCMD_LIST=("sh" "-c" "${RUNCMD}")
    RUNCMD_LIST_STR=$(printf '%s\n' "${RUNCMD_LIST[@]}" | jq -R . | jq -s .)

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
            "devices": ['$DEVICE_PROP_STR'],
            "sharedMemorySize": '${MEMORY_PER_NODE}',
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

    # Register and submit job
    response_reg_job_def=$(aws batch register-job-definition \
        --job-definition-name $JOB_DEFINITION_NAME \
        --type $JOB_TYPE \
        --scheduling-priority $JOB_PRIORITY \
        --node-properties "$NODE_PROP_STR" \
        --region $REGION \
        --output table)
    echo "response_reg_job_def: $response_reg_job_def"
    echo "Registered job definition $JOB_DEFINITION_NAME"

    sleep 10

    echo "Submitting job"
    aws batch submit-job --job-name $JOB_NAME --job-queue $JOB_QUEUE --job-definition $JOB_DEFINITION_NAME --share-identifier $SHARE_IDENTIFIER --region $REGION
    echo "Submitted job $JOB_NAME"
done