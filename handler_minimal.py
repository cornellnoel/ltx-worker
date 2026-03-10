import runpod
print("HANDLER ALIVE")
runpod.serverless.start({"handler": lambda job: {"result": "hello from ltx"}})
