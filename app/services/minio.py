from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error, InvalidResponseError
import os
load_dotenv()


class MINIOservice:
    def __init__(self):
        self.Client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
        )
         
    
    def upload_file(self, file_path: str, object_name: str):
        try:
            self.Client.fput_object(
                bucket_name=os.getenv("MINIO_BUCKET"),
                object_name=object_name,
                file_path=file_path,
            )
            return(f"File {file_path} uploaded as {object_name} in bucket {os.getenv('MINIO_BUCKET')}.")
        except S3Error as e:
            return(f"Error occurred: {e}")
    
    def download_file(self, object_name: str, file_path: str):
        self.Client.fget_object(
            bucket_name=os.getenv("MINIO_BUCKET"),
            object_name=object_name,
            file_path=file_path,
        )
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()