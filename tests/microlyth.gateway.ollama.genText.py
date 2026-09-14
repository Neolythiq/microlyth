from microlyth.src.gateway import OllamaTextGenClient

if __name__ == "__main__":
    client = OllamaTextGenClient()
    task = "Provide a cookie recipe."
    response = client.SendRequest(task)
    print(response)