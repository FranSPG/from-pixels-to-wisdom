import gc
import requests
import torch.cuda

from PIL import Image
from tritonclient.utils import *
import tritonclient.http as httpclient

from fastapi import FastAPI

app = FastAPI()

client = httpclient.InferenceServerClient(url="localhost:8000")


@app.post('/doc_classification/')
def doc_image_classification(request: str):
    model_name = 'python_donut_image_classification'

    # Inputs
    image = np.asarray(Image.open(requests.get(request, stream=True).raw)).astype(np.float16)
    image = np.expand_dims(image, axis=0)
    # Set Inputs
    input_tensors = [
        httpclient.InferInput("pixel_values", image.shape, datatype="FP16")
    ]
    input_tensors[0].set_data_from_numpy(image)

    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]
    # Query
    query_response = client.infer(model_name=model_name,
                                  inputs=input_tensors,
                                  outputs=outputs)

    # Output
    result = query_response.as_numpy('output').tolist()

    # client.close()
    # report_gpu()
    return result


@app.post('/doc_parsing/')
def doc_image_parsing(request: str):
    model_name = 'python_donut_doc_parsing'

    image = np.asarray(Image.open(requests.get(request, stream=True).raw)).astype(np.float16)
    image = np.expand_dims(image, axis=0)
    # Set Inputs
    input_tensors = [
        httpclient.InferInput("pixel_values", image.shape, datatype="FP16")
    ]
    input_tensors[0].set_data_from_numpy(image)

    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]
    # Query
    query_response = client.infer(model_name=model_name,
                                  inputs=input_tensors,
                                  outputs=outputs)

    # Output
    result = query_response.as_numpy('output').tolist()

    return result


@app.post('/question_answering/')
def doc_question_answering(request: str, question: str):
    model_name = 'python_donut_question_answering'
    image = np.asarray(Image.open(requests.get(request, stream=True).raw)).astype(np.float16)
    image = np.expand_dims(image, axis=0)

    # question = str.encode(question)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("pixel_values", image.shape, datatype="FP16"),
        httpclient.InferInput("question", [1, 1], datatype="BYTES"),
    ]

    input_tensors[0].set_data_from_numpy(image)

    input_data = np.array([str(x).encode('utf-8') for x in [question]],
                          dtype=np.object_)
    input_data = input_data.reshape((1, 1))
    input_tensors[1].set_data_from_numpy(input_data)

    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]
    # Query
    query_response = client.infer(model_name=model_name,
                                  inputs=input_tensors,
                                  outputs=outputs)

    # Output
    result = query_response.as_numpy('output').tolist()

    return result


