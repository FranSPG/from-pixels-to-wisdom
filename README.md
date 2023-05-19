# from-pixels-to-wisdom
Welcome my series, "From Pixels to Wisdom", where we'll explore the exciting world of Computer Vision and Machine Learning, along with their potential to unlock insights and knowledge from data. In this series, we'll dive into cutting-edge techniques and best practices for deploying them in production environments and at scale successfully.
There is a positive transformation occurring in machine learning, where the emphasis is shifting from merely achieving functional models to ensuring they align with the organizartion's requirements. Through my experience with ML systems over the past years, I have discovered that effectively creating, delivering, and operating ML models in an efficient, repeatable, and scalable manner is a callenging endeavor.

# How to run it locally

### Server
'''docker build -f Dockerfile_server -t triton_server .'''
'''docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/model_repository triton_server'''

### Client

'''docker build -f Dockerfile_client -t client_triton .'''
'''docker run -it --net=host -v ${PWD}:/workspace/ client_triton'''

Once it's running go to 127.0.0.1:8080/docs