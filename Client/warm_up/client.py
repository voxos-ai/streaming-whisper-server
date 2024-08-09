import requests




res = requests.post("http://127.0.0.1:6700/load-model/",json={
    "model-name":"tiny",
    "mode":"auto"
})

print(res.status_code)
model = res.json()["model-id"]
print(model)