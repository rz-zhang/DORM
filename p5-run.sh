REGION=us-west-2
CONFIG_NAME=1120-batch-dorm-mix-4node-p5
export GPFS="/dorm-workspace"

declare -A queues
queues[FS-P5_48XL-Training-us-west-2b]=p5.48xlarge
#queues[FS-P4DE_24XL-Training-us-west-2a]=p4de.24xlarge

for queue in "${!queues[@]}"; do
    JOB_QUEUE=$queue
    INSTANCE_TYPE=${queues[$queue]}

    JOB_PRIORITY=100
    SHARE_IDENTIFIER=NeoTeam # NeoTeam, Smaller4NodeTraining, Evaluation

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

    RUNCMD="
        # Function to log messages with timestamp
        log_message() {
            echo \"\$(date '+%Y-%m-%d %H:%M:%S') - \$1\"
        }

        # Function to check and wait for filesystem mount
        check_fs() {
            local max_wait=300
            local wait_time=0
            log_message \"Checking filesystem mount...\"

            while [ ! -d \"${GPFS}\" ]; do
                if [ \$wait_time -ge \$max_wait ]; then
                    log_message \"ERROR: Timeout waiting for filesystem mount after \${max_wait} seconds\"
                    exit 1
                fi
                log_message \"Waiting for filesystem mount... (\${wait_time} seconds elapsed)\"
                sleep 10
                wait_time=\$((wait_time + 10))
            done
            log_message \"Filesystem mounted successfully\"
        }

        # Stage 1: Check filesystem and create working directory
        check_fs && \
        mkdir -p ${GPFS} && \
        cd ${GPFS} && \

        # Stage 2: Code download and synchronization
        if [ \"\${AWS_BATCH_JOB_NODE_INDEX:-0}\" = \"0\" ]; then
            # Main node: Download code and prepare environment
            log_message \"Node 0: Starting code download...\" && \
            rm -f /tmp/download_complete && \
            aws s3 sync s3://shopqa-users/ronzhi/Mistral-NeMo-12B-Instruct ./Mistral-NeMo-12B-Instruct && \
            aws s3 sync s3://shopqa-users/ronzhi/12b ./12b && \

            # Verify downloads
            if [ -d \"./12b/NeMo-Aligner\" ]; then
                touch /tmp/download_complete && \
                log_message \"Node 0: Downloads completed and verified\"
            else
                log_message \"ERROR: Node 0: Download verification failed\"
                exit 1
            fi
        else
            # Worker nodes: Wait for code download
            log_message \"Node \${AWS_BATCH_JOB_NODE_INDEX}: Waiting for code downloads...\"
            max_wait=1800  # 30 minutes timeout
            wait_time=0
            while [ ! -f /tmp/download_complete ]; do
                if [ \$wait_time -ge \$max_wait ]; then
                    log_message \"ERROR: Timeout waiting for downloads after \${max_wait} seconds\"
                    exit 1
                fi
                log_message \"Node \${AWS_BATCH_JOB_NODE_INDEX}: Still waiting... (\${wait_time} seconds elapsed)\"
                sleep 30
                wait_time=\$((wait_time + 30))
            done
            log_message \"Node \${AWS_BATCH_JOB_NODE_INDEX}: Downloads confirmed complete\"
        fi && \

        # Stage 3: Node synchronization
        if [ \"\${AWS_BATCH_JOB_NODE_INDEX:-0}\" = \"0\" ]; then
            # Main node: Wait for all worker nodes to be ready
            log_message \"Node 0: Initiating node synchronization\" && \
            rm -f /tmp/nodes_ready && \
            touch /tmp/node0_ready && \

            # Wait for all worker nodes
            for i in \$(seq 1 $((NUM_NODES-1))); do
                while [ ! -f \"/tmp/node\${i}_ready\" ]; do
                    log_message \"Node 0: Waiting for node \${i} ready signal...\"
                    sleep 20
                done
                log_message \"Node 0: Detected node \${i} is ready\"
            done

            touch /tmp/nodes_ready
            log_message \"Node 0: All nodes reported ready\"
        else
            # Worker nodes: Signal ready and wait for all-clear
            node_id=\${AWS_BATCH_JOB_NODE_INDEX}
            log_message \"Node \${node_id}: Signaling ready state\"
            touch \"/tmp/node\${node_id}_ready\"

            while [ ! -f /tmp/nodes_ready ]; do
                log_message \"Node \${node_id}: Waiting for all nodes ready signal...\"
                sleep 20
            done
            log_message \"Node \${node_id}: Received all-clear signal\"
        fi && \

        # Stage 4: Network connectivity check
        if [ \"\${AWS_BATCH_JOB_NODE_INDEX:-0}\" = \"0\" ]; then
            # Main node: Verify network connectivity
            log_message \"Node 0: Starting network connectivity check\" && \
            rm -f /tmp/network_ready && \

            # Get all node IPs
            all_ips=(\$(hostname -i))
            for i in \$(seq 1 $((NUM_NODES-1))); do
                ip_index=\$((i+1))
                target_ip=\${all_ips[\$ip_index]}
                retry_count=0
                max_retries=5

                while ! ping -c 1 \$target_ip &>/dev/null; do
                    if [ \$retry_count -ge \$max_retries ]; then
                        log_message \"ERROR: Failed to establish connectivity with node \${i} (\${target_ip})\"
                        exit 1
                    fi
                    log_message \"Node 0: Retrying connection to node \${i} (\${target_ip})...\"
                    sleep 10
                    retry_count=\$((retry_count + 1))
                done
                log_message \"Node 0: Successfully connected to node \${i} (\${target_ip})\"
            done

            touch /tmp/network_ready
            log_message \"Node 0: Network connectivity verified for all nodes\"
        else
            # Worker nodes: Wait for network verification
            while [ ! -f /tmp/network_ready ]; do
                log_message \"Node \${AWS_BATCH_JOB_NODE_INDEX}: Waiting for network verification...\"
                sleep 10
            done
            log_message \"Node \${AWS_BATCH_JOB_NODE_INDEX}: Network connectivity confirmed\"
        fi && \

        # Stage 5: Environment setup and training initialization
        log_message \"Setting up NCCL environment variables\" && \
        export NCCL_DEBUG=INFO && \
        export NCCL_IB_DISABLE=0 && \
        export NCCL_NET_GDR_LEVEL=2 && \
        export NCCL_P2P_DISABLE=0 && \
        export NCCL_SOCKET_IFNAME=eth0 && \
        export NCCL_MIN_NRINGS=8 && \
        export NCCL_TREE_THRESHOLD=0 && \
        export NCCL_BUFFSIZE=2097152 && \

        # Stage 6: Training execution with retry mechanism
        cd 12b/NeMo-Aligner && \
        chmod +x *.sh && \
        log_message \"Starting training execution\" && \

        MAX_RETRIES=3
        retry_count=0
        while [ \$retry_count -lt \$MAX_RETRIES ]; do
            if [ \$retry_count -gt 0 ]; then
                log_message \"Retry attempt \$((retry_count+1))/\$MAX_RETRIES\"
                sleep 30
            fi

            ${DDP_ENVS} ./shopqa-dorm-mix-train.sh

            if [ \$? -eq 0 ]; then
                log_message \"Training completed successfully\"
                break
            else
                log_message \"Training attempt failed\"
                retry_count=\$((retry_count + 1))
                if [ \$retry_count -ge \$MAX_RETRIES ]; then
                    log_message \"ERROR: Training failed after \$MAX_RETRIES attempts\"
                    exit 1
                fi
            fi
        done"

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