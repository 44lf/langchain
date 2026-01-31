from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
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
        # 验证必要的环境变量
        required_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET"]
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"必需的环境变量 {var} 未设置")
        
        # 确保存储桶存在
        self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """检查并创建存储桶（如果不存在）"""
        bucket_name = os.getenv("MINIO_BUCKET")
        if not self.Client.bucket_exists(bucket_name):
            try:
                self.Client.make_bucket(bucket_name)
                print(f"存储桶 '{bucket_name}' 创建成功!")
            except S3Error as e:
                print(f"创建存储桶时出错: {e}")
                raise
        else:
            print(f"存储桶 '{bucket_name}' 已存在.")
    
    def list_objects(self, prefix=""):
        """列出存储桶中的对象"""
        try:
            bucket_name = os.getenv("MINIO_BUCKET")
            objects = self.Client.list_objects(bucket_name, prefix=prefix, recursive=True)
            object_list = [obj.object_name for obj in objects]
            return object_list
        except S3Error as e:
            print(f"列出对象时出错: {e}")
            return []
    
    def upload_file(self, file_path: str, object_name: str):
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return f"错误：本地文件 {file_path} 不存在."
            
            bucket_name = os.getenv("MINIO_BUCKET")
            
            # 确保存储桶存在
            if not self.Client.bucket_exists(bucket_name):
                self.Client.make_bucket(bucket_name)
            
            self.Client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
            )
            return f"文件 {file_path} 已上传为 {object_name} 到存储桶 {bucket_name}."
        except S3Error as e:
            return f"发生S3错误：{e}"
        except Exception as e:
            return f"发生未知错误：{e}"
    
    def download_file(self, object_name: str, file_path: str):
        try:
            bucket_name = os.getenv("MINIO_BUCKET")
            
            # 确保存储桶存在
            if not self.Client.bucket_exists(bucket_name):
                return f"错误：存储桶 {bucket_name} 不存在."
            
            # 检查对象是否存在
            try:
                self.Client.stat_object(bucket_name, object_name)
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    available_objects = self.list_objects()
                    if available_objects:
                        return f"错误：对象 {object_name} 在存储桶 {bucket_name} 中不存在。可用对象: {available_objects}"
                    else:
                        return f"错误：对象 {object_name} 在存储桶 {bucket_name} 中不存在。存储桶中没有任何对象。"
                else:
                    return f"S3错误：{e}"
            
            # 从MinIO下载文件到本地
            self.Client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
            )
            
            # 尝试按文本读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # 如果是二进制文件，返回文件路径
                return f"文件已下载到: {file_path}"
        except S3Error as e:
            return f"S3错误：{e}"
        except Exception as e:
            return f"未知错误：{e}"