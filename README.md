# To launch the Triton inference server with Docker locally

1. Run a docker container with the tritonserver image. Make sure you are inside the triton folder.
  > docker run --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3
2. Inside the container, install all necessary packages (refer to requirements.txt).
3. Launch the triton inference server:
  > tritonserver --model-repository=/models --log-verbose=1


# Launch Chat UI locally and interface with the Triton server

1. Install flask and other dependency packages.
2. Run the python script app.py. The server should be running on: http://127.0.0.1:5000, otherwise amend the index.html script to use the correct API.
3. Open index.html using a web browser and start interacting with the LLM.


# Launch Locust locally to perform load testing on the Triton server

1. Install Locust on your machine.
2. Run the locust application: 
  > locust -f script.py
3. Start to increase or decrease the load.


# Deploy in a Kubernetes Cluster

1) Setup a cluster, such as using Minikube.
2) There is deployment.yaml file in Triton folder, run:
   > kubectl apply -f deployment.yaml -f service.yaml  
   It should create triton deployment and service, make sure you mount the folder where the model_repository is as /mnt/data. 
3) Once the server is running, we need to forward the port used by the pod to be accessible as localhost from our computer. Run:
  > kubectl port-forward svc/triton 8000:8000
    We can perform health check using curl command:
    > curl -X POST localhost:8000/v2/repository/index
4) Take note of the triton service IP address, we need to use this for our Chat UI and Locust.
  > kubectl get svc
5) Under chatui folder, update the env variable TRITON_SERVER_URL to be using the IP address obtained from step 4 above. Then run:
  > kubectl apply -f deployment.yaml
  It will create a Chat UI deployment as well as service. We also need to fo port-forwarding:
  > kubectl port-forward svc/chatui 5000:5000
6) Open index.html in a browser, a dialog box should be available and we can query the LLM.
7) To setup Locust, go to locust folder and run:
  > kubectl apply -f scripts-cm.yaml -f master-deployment.yaml -f service.yaml -f slave-deployment.yaml
  The Locust server is listening on port 8089, we need to forward this port as well:
  > kubectl port-forward svc/locust-master 8089:8089
8) We can then access the Locust dashboard and perform load testing.
