import boto3
import botocore
import urllib3
from datetime import datetime


from .config import Config


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Клас описывает взаимодействие (только чтение) с s3 хранилищем
class Hub:
    _instance = None
    
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self):
        s3config = botocore.config.Config(
            connect_timeout = 5,
            read_timeout = 5,
            retries = {'max_attempts': 3}
        )

        self.__config = Config()
        self.__s3 = boto3.client(
          's3',
          aws_access_key_id = self.__config['AWS_ACCESS_KEY_ID'],
          aws_secret_access_key = self.__config['AWS_SECRET_ACCESS_KEY'],
          endpoint_url = self.__config['S3_ENDPOINT_URL'],
          verify = False,
          config = s3config
        )

    def last_modified(self, s3_path: str)->datetime:
        """Возвращает дату последней модификации файла в хранилище"""
        return self.__s3.head_object(Bucket=self.__config['BUCKET_NAME'], Key=s3_path)['LastModified']

    def download(self,  s3_path: str, local_path: str)->None:
        """Скачивает файл с хранилища"""
        self.__s3.download_file(self.__config['BUCKET_NAME'], s3_path, local_path)

    def is_exist(self, s3_path: str)->bool:
        """Проверяет наличие файла в хранилище"""
        try:
            self.__s3.head_object(Bucket=self.__config['BUCKET_NAME'], Key=s3_path)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                raise e
