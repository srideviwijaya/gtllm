from locust import HttpUser, task, between

class llamav2user(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def post_infer(self):
        response = self.client.post("/v2/models/llamav2/infer", json={"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["hello, how are you?"]}]})
        print(response.json())

