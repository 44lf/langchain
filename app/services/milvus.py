from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from dotenv import load_dotenv  # 缺少import
import os

load_dotenv()  # 应在类外调用

class MILVUSService:
    def __init__(self):
        self.milvus_alias = "default"
        self.db_name = os.getenv("MILVUS_DEFAULT_DB", "langchain_db")
        self.collection_name = os.getenv("MILVUS_DEFAULT_COLLECTION", "langchain_collection")
    
    def setup_milvus(self):
        """初始化Milvus连接和环境"""
        connections.connect(
            alias=self.milvus_alias,  # 使用self
            host='localhost',
            port='19530'
        )

        # 清理旧环境
        if self.db_name in db.list_database(using=self.milvus_alias):  # 使用self
            db.using_database(self.db_name, using=self.milvus_alias)
            for coll in utility.list_collections(using=self.milvus_alias):
                Collection(coll, using=self.milvus_alias).drop()
            db.drop_database(self.db_name, using=self.milvus_alias)

        # 创建新数据库
        db.create_database(self.db_name, using=self.milvus_alias)
        db.using_database(self.db_name, using=self.milvus_alias)
        return True

    def create_collection(self, metric_type, dim=128):  # 缺少self和dim参数
        """创建指定相似度度量的集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="label", dtype=DataType.INT64),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=100)
        ]

        schema = CollectionSchema(fields, description=f"使用:{metric_type}的集合")
        collection = Collection(
            name=f"{self.collection_name}_{metric_type.lower()}",  # 使用self
            schema=schema,
            using=self.milvus_alias
        )

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {"nlist": 100}
        }
        collection.create_index("vector", index_params)
        return collection
    
    def insert_vector(self, collection, vectors, labels, descs):
        """插入向量数据"""
        mr = collection.insert([vectors, labels, descs])
        collection.flush()
        return mr
