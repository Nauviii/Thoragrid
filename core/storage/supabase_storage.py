"""Upload files to Supabase Storage and generate time-limited signed URLs."""

from supabase import create_client, Client

from config.settings import settings

_SIGNED_URL_EXPIRY_SECONDS = 3600  # X-ray images are sensitive; signed, not permanent public links

_client: Client | None = None


def _get_client() -> Client:
    """Lazily create and cache the Supabase client."""
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client


def create_signed_url(bucket: str, path: str, expiry_seconds: int = _SIGNED_URL_EXPIRY_SECONDS) -> str:
    """Return a fresh time-limited signed URL for an object already in storage."""
    signed = _get_client().storage.from_(bucket).create_signed_url(path, expiry_seconds)
    return signed["signedURL"]


def upload_and_sign(bucket: str, path: str, data: bytes, content_type: str = "image/png") -> str:
    """Upload bytes to a bucket path and return a time-limited signed URL."""
    client = _get_client()
    client.storage.from_(bucket).upload(
        path, data, {"content-type": content_type, "upsert": "true"}
    )
    return create_signed_url(bucket, path)


def xray_path(image_hash: str) -> str:
    """Deterministic storage path for a normalized X-ray upload; matches api/routes/image.py."""
    return f"{image_hash}.png"


def heatmap_path(interaction_id: str, condition: str) -> str:
    """Deterministic storage path for one condition's GradCAM heatmap; matches api/routes/image.py."""
    return f"{interaction_id}/{condition}.png"