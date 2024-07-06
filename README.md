To launch the Triton inference server

1. Run a docker container with the tritonserver image. Make sure you are inside the triton folder.
    docker run --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3
2. Inside the container, install all necessary packages (refer to requirements.txt).
3. Launch the triton inference server: tritonserver --model-repository=/models --log-verbose=1


Chat UI

1. Install flask and other dependency packages.
2. Run the python script app.py. The server should be running on: http://127.0.0.1:5000, otherwise amend the index.html script to use the correct API.
3. Open index.html using a web browser and start interacting with the LLM.

Locust

1. Install Locust.
2. Run the locust application: locust -f script.py.
3. Start to increase or decrease the load.
