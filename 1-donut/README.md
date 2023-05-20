# #1 From Pixels To Wisdom | OCR-free Document Understanding Transformer

Join me today as we explore the exciting capabilities of Donut, a powerful tool designed for document understanding tasks. The primary objective of this post is to introduce ourselves to both the topic and the tool and promptly implement the models in a production-like environment using Triton Inference Server to perform inference using a readily available model swiftly.

# How to run it locally?

- Note that build times are pretty significant

## Docker Compose

Docker Compose is the preferable method of managing your local development environment. Make sure to install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) to use the examples below.

### Start Docker Compose

To **start** your local development environment, execute the following command:

```bash
docker compose up -d
```

Please note that you must allow time for Triton Server to download necessary models. To ensure that Triton is ready for inference, execute the following command:

```bash
curl -v localhost:8000/v2/health/ready
```

If Triton Inference Server is ready, you will receive `< HTTP/1.1 200 OK` response.

### Stop Docker Compose

To **stop** your local development environment, execute the following command:

```bash
docker compose down
```

## Docker

If you prefer to set up your local development environment manually, please install [Docker](https://docs.docker.com/get-docker/) and follow the instructions below.

### Triton Server

First, you need to build Triton Server:

```bash
docker build -f Dockerfile_server -t fptw/triton-server .
```

Second, you need to run Triton Server:

```bash
docker run --rm --gpus=all --shm-size=3gb -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/model_repository/ fptw/triton-server
```

### Triton Client

Third, you need to build Triton Client:

```bash
docker build -f Dockerfile_client -t fptw/triton-client .
```

Finally, you need to run Triton Client:

```bash
docker run --rm --network host fptw/triton-client
```

### Usage

To use the environment, please visit http://localhost:8080/docs
