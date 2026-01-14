# Makefile for building Docker images with specific CUDA versions

CUDA_REPO = nvidia/cuda

OUTPUT_REPO = huyu/cuda-dev

CUDA_VERSION_11_8 = 11.8-ubuntu22.04
CUDA_DEV_IMAGE_11_8 = 11.8.0-cudnn8-devel-ubuntu22.04
CUDA_RUNTIME_IMAGE_11_8 = 11.8.0-cudnn8-runtime-ubuntu22.04

CUDA_VERSION_12_4 = 12.4-ubuntu22.04
CUDA_DEV_IMAGE_12_4 = 12.4.1-cudnn-devel-ubuntu22.04
CUDA_RUNTIME_IMAGE_12_4 = 12.4.1-cudnn-runtime-ubuntu22.04

CUDA_VERSION_12_8 = 12.8-ubuntu22.04
CUDA_DEV_IMAGE_12_8 = 12.8.1-cudnn-devel-ubuntu22.04
CUDA_RUNTIME_IMAGE_12_8 = 12.8.1-cudnn-runtime-ubuntu22.04

DEV_DOCKERFILE = docker/Dockerfile.dev

.PHONY: cuda-dev/11.8 cuda-dev/12.4 cuda-dev/12.8 dev all

cuda-dev/11.8:
	docker build \
		--progress=plain \
		-t $(OUTPUT_REPO):$(CUDA_VERSION_11_8) \
		--build-arg "BASE_IMAGE=$(CUDA_REPO):$(CUDA_DEV_IMAGE_11_8)" \
		-f $(DEV_DOCKERFILE) .

cuda-dev/12.4:
	docker build \
		--progress=plain \
		-t $(OUTPUT_REPO):$(CUDA_VERSION_12_4) \
		--build-arg "BASE_IMAGE=$(CUDA_REPO):$(CUDA_DEV_IMAGE_12_4)" \
		-f $(DEV_DOCKERFILE) .

cuda-dev/12.8:
	docker build \
		--progress=plain \
		-t $(OUTPUT_REPO):$(CUDA_VERSION_12_8) \
		--build-arg "BASE_IMAGE=$(CUDA_REPO):$(CUDA_DEV_IMAGE_12_8)" \
		-f $(DEV_DOCKERFILE) .

dev: cuda-dev/11.8 cuda-dev/12.4 cuda-dev/12.8

all: dev