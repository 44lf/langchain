from pymilvus import connections
import os
load_dotenv()

class MILVUSService:
    def __init__(self):
        self.alias = os.getenv("MILVUS_ALIAS", "default")
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")


    def connect(self):
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        return true