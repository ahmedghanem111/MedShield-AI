
from redis import Redis
from rq import Queue

from app.core.config import settings


redis_connection = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=False,
    socket_connect_timeout=5,
    socket_timeout=5,
    health_check_interval=30,
)

audit_queue = Queue(
    name="medical_audits",
    connection=redis_connection,
    default_timeout=600,
)
