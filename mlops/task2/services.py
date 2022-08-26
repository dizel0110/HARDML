import os
import time
import string
import random
import signal
import threading
import redis


def gen_replica_name(replica_prefix: str = 'web_app_replica_',
                     replica_id_len: int = 10) -> str:
    chars = string.ascii_lowercase + string.digits
    replica_id = ''.join([random.choice(chars) 
                          for _ in range(replica_id_len)])
    return replica_prefix + replica_id


class ServiceDiscovery(object):
    def __init__(self,
                 redis_host: str,
                 redis_port: str,
                 redis_password: str,
                 replicas_key: str = 'web_app', ) -> None:
        super().__init__()
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        
        self.replicas_key = replicas_key

    def init_redis(self) -> redis.Redis:
        return redis.Redis(host=self.redis_host,
                           port=self.redis_port,
                           password=self.redis_password,
                           decode_responses=True, )

    def register(self, name: str, parameters: dict):
        with self.init_redis() as redis:
            redis.lpush(self.replicas_key, name)
            redis.hmset(name, parameters)

    def unregister(self, name: str):
        with self.init_redis() as redis:
            redis.lrem(self.replicas_key, 1, name)
            redis.delete(name)

        def exit():
            time.sleep(15)
            os.kill(os.getpid(), signal.SIGKILL)

        thread = threading.Thread(target=exit)
        thread.start()
