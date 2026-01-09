# Makefile for building Docker images with specific CUDA versions

CUDA_VERSION_11_8 = 11.8-ubuntu22.04
CUDA_BASE_IMAGE_11_8 = nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

CUDA_VERSION_12_4 = 12.4-ubuntu22.04
CUDA_BASE_IMAGE_12_4 = nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

CUDA_VERSION_12_8 = 12.8-ubuntu22.04
CUDA_BASE_IMAGE_12_8 = nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04


DOCKERFILE = src/Dockerfile

.PHONY: cuda/11.8 cuda/12.4 cuda/12.8 all

cuda/11.8:
	docker build \
		--progress=plain \
		-t huyu/cuda:$(CUDA_VERSION_11_8) \
		--build-arg "BASE_IMAGE=$(CUDA_BASE_IMAGE_11_8)" \
		-f $(DOCKERFILE) .

cuda/12.4:
	docker build \
		--progress=plain \
		-t huyu/cuda:$(CUDA_VERSION_12_4) \
		--build-arg "BASE_IMAGE=$(CUDA_BASE_IMAGE_12_4)" \
		-f $(DOCKERFILE) .

cuda/12.8:
	docker build \
		--progress=plain \
		-t huyu/cuda:$(CUDA_VERSION_12_8) \
		--build-arg "BASE_IMAGE=$(CUDA_BASE_IMAGE_12_8)" \
		-f $(DOCKERFILE) .

all: cuda/11.8 cuda/12.4
