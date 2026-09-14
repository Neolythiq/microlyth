from microlyth.src.gateway import LocalBGEM3MClient

if __name__ == "__main__":
    client = LocalBGEM3MClient(endPoint=r"C:\\Users\\jerem\\Documents\\Repository\\Trilyth-test\\models\\bge-m3")
    text = "Sample text for embedding"
    embedding = client.GetEmbedding(text)
    print(embedding)