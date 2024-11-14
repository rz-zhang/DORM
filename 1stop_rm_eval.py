#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import os
import logging
import threading
import queue
from typing import Optional
from pathlib import Path

class DockerRewardModelEvaluator:
    def __init__(
        self,
        workspace_path: str,
        results_path: str,
        model_path: str,
        input_file: str,
        output_file: str,
        port: int = 1424,
        num_nodes: int = 1,
        num_devices: int = 8,
        tensor_parallel_size: int = 8,
        pipeline_parallel_size: int = 1,
        network_name: str = "rm_network",
        docker_image: str = "nemo-aligner-image-12b",
        log_file: Optional[str] = "reward_eval.log"
    ):
        self.workspace_path = Path(workspace_path)
        self.results_path = Path(results_path)
        self.model_path = model_path
        self.input_file = input_file
        self.output_file = output_file
        self.port = port
        self.num_nodes = num_nodes
        self.num_devices = num_devices
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.network_name = network_name
        self.docker_image = docker_image
        self.server_container = None
        self.client_container = None

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 用于存储容器输出的队列
        self.server_output_queue = queue.Queue()
        self.client_output_queue = queue.Queue()

    def _stream_container_output(self, container_name: str, output_queue: queue.Queue):
        """实时流式读取容器输出"""
        try:
            process = subprocess.Popen(
                f"docker logs -f {container_name}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:  # 只处理非空行
                    output_queue.put(line)
                    print(f"[{container_name}] {line}")

            process.stdout.close()
            process.wait()
        except Exception as e:
            self.logger.error(f"Error in output streaming for {container_name}: {str(e)}")

    def setup_docker_network(self):
        """创建Docker网络"""
        try:
            # 检查网络是否已存在
            result = subprocess.run(
                f"docker network ls | grep {self.network_name}",
                shell=True,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                self.logger.info(f"Creating Docker network: {self.network_name}")
                subprocess.run(
                    f"docker network create {self.network_name}",
                    shell=True,
                    check=True
                )
            else:
                self.logger.info(f"Docker network {self.network_name} already exists")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to setup Docker network: {str(e)}")
            raise

    def start_server_container(self):
        """启动服务器容器"""
        try:
            docker_cmd = f"""
            docker run -d --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v {self.workspace_path}:/workspace/NeMo-Aligner \
            -v {self.workspace_path}/tmp:/tmp \
            -v {self.workspace_path}/var/tmp:/var/tmp \
            -v {self.workspace_path}/log:/var/log \
            -v {self.results_path}:/results \
            --network {self.network_name} \
            --name rm_service \
            -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
            -e HF_TOKEN='{os.getenv("HF_TOKEN")}' \
            -w /workspace/NeMo-Aligner \
            {self.docker_image}
            """

            # 启动容器
            self.server_container = subprocess.run(
                docker_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            # 开始监控服务器输出
            server_output_thread = threading.Thread(
                target=self._stream_container_output,
                args=("rm_service", self.server_output_queue)
            )
            server_output_thread.daemon = True
            server_output_thread.start()

            # 在容器中启动服务
            serve_cmd = f"""
            docker exec rm_service bash -c "cd /workspace/NeMo-Aligner && \
            python ./examples/nlp/gpt/serve_reward_model.py \
            rm_model_file={self.model_path} \
            trainer.num_nodes={self.num_nodes} \
            trainer.devices={self.num_devices} \
            ++model.tensor_model_parallel_size={self.tensor_parallel_size} \
            ++model.pipeline_model_parallel_size={self.pipeline_parallel_size} \
            inference.port={self.port}"
            """

            self.logger.info("Starting reward model server...")
            self.server_process = subprocess.Popen(
                serve_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # 等待服务就绪
            self._wait_for_server()

        except Exception as e:
            self.logger.error(f"Failed to start server container: {str(e)}")
            self.cleanup()
            raise

    def _wait_for_server(self, timeout=300, check_interval=5):
        """等待服务器就绪"""
        self.logger.info("Waiting for server to be ready...")
        start_time = time.time()

        success_markers = [
            "Started HTTPService at",
            "successfully loaded 'reward_model'"
        ]
        found_markers = {marker: False for marker in success_markers}

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Server failed to start within {timeout} seconds")

            # 从输出队列中检查标记
            try:
                while True:
                    line = self.server_output_queue.get_nowait()
                    for marker in success_markers:
                        if marker in line and not found_markers[marker]:
                            found_markers[marker] = True
                            self.logger.info(f"Found marker: {marker}")
            except queue.Empty:
                pass

            # 检查是否所有标志都已找到
            if all(found_markers.values()):
                self.logger.info("Server is fully ready")
                break

            time.sleep(check_interval)

    def run_inference(self):
        """运行推理容器"""
        try:
            docker_cmd = f"""
            docker run -d --rm --gpus all --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v {self.workspace_path}:/workspace/NeMo-Aligner \
            -v {self.workspace_path}/tmp:/tmp \
            -v {self.workspace_path}/var/tmp:/var/tmp \
            -v {self.workspace_path}/log:/var/log \
            -v {self.results_path}:/results \
            --network {self.network_name} \
            --name rm_client \
            -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
            -w /workspace/NeMo-Aligner \
            {self.docker_image}
            """

            # 启动客户端容器
            self.client_container = subprocess.run(
                docker_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            # 开始监控客户端输出
            client_output_thread = threading.Thread(
                target=self._stream_container_output,
                args=("rm_client", self.client_output_queue)
            )
            client_output_thread.daemon = True
            client_output_thread.start()

            # 运行推理
            inference_cmd = f"""
            docker exec rm_client bash -c "cd /workspace/NeMo-Aligner && \
            python ./examples/nlp/data/steerlm/run_rb.py \
            --input-file={self.input_file} \
            --output-file={self.output_file} \
            --host=rm_service \
            --port={self.port}"
            """

            self.logger.info("Starting inference...")
            result = subprocess.run(
                inference_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info("Inference completed successfully")
            return result.stdout

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Inference failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during inference: {str(e)}")
            raise

    def cleanup(self):
        """清理Docker容器和网络"""
        for container in ['rm_service', 'rm_client']:
            try:
                subprocess.run(
                    f"docker stop {container}",
                    shell=True,
                    capture_output=True,
                    check=False
                )
            except Exception as e:
                self.logger.warning(f"Error stopping container {container}: {str(e)}")

    def __enter__(self):
        self.setup_docker_network()
        self.start_server_container()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

def main():
    # 设置信号处理
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if hasattr(main, "evaluator"):
            main.evaluator.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 配置参数
        config = {
            "workspace_path": "/fsx-Training/home/$USER/12b/NeMo-Aligner",  # 替换 $USER 为实际用户名
            "results_path": "/fsx-Training/home/$USER/west_results_model",  # 替换 $USER 为实际用户名
            "model_path": "/results/1025_west_2a_1node_hs2_20k_3epoch_lr3e6/checkpoints/megatron_gpt.nemo",
            "input_file": "data/rewardbench/rewardbench.jsonl",
            "output_file": "data/rewardbench/1112_pred_12b_oh_40k_lr3e6_3epoch_1node.jsonl",
            "log_file": "reward_model_eval.log"
        }

        print("Starting reward model evaluation...")
        print(f"Log file: {config['log_file']}")

        with DockerRewardModelEvaluator(**config) as evaluator:
            main.evaluator = evaluator
            results = evaluator.run_inference()
            print("\nFinal Results:")
            print(results)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()