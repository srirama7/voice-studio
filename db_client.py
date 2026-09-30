"""
Firebase Firestore & Storage Database Client for Voice Studio.

Provides unified persistent storage for voice profiles, job state checkpoints,
and binary audio/visual assets in Firebase project `voiceai-6c2e1`.
Supports `firebase_admin` SDK, REST API fallback, and offline local cache persistence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import requests

from config import Config, default_config

logger = logging.getLogger("db_client")
logger.setLevel(logging.INFO)

# Try importing firebase_admin
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    _HAS_FIREBASE_ADMIN = True
except ImportError:
    _HAS_FIREBASE_ADMIN = False


class FirestoreRESTHelper:
    """Helper to serialize/deserialize standard Python dicts to/from Firestore REST format."""
    
    @staticmethod
    def dict_to_firestore(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Python dict to Firestore REST Document fields structure."""
        fields = {}
        for key, val in data.items():
            fields[key] = FirestoreRESTHelper._val_to_firestore(val)
        return {"fields": fields}

    @staticmethod
    def _val_to_firestore(val: Any) -> Dict[str, Any]:
        if val is None:
            return {"nullValue": None}
        elif isinstance(val, bool):
            return {"booleanValue": val}
        elif isinstance(val, int):
            return {"integerValue": str(val)}
        elif isinstance(val, float):
            return {"doubleValue": val}
        elif isinstance(val, str):
            return {"stringValue": val}
        elif isinstance(val, list):
            return {"arrayValue": {"values": [FirestoreRESTHelper._val_to_firestore(v) for v in val]}}
        elif isinstance(val, dict):
            sub_fields = {k: FirestoreRESTHelper._val_to_firestore(v) for k, v in val.items()}
            return {"mapValue": {"fields": sub_fields}}
        else:
            return {"stringValue": str(val)}

    @staticmethod
    def firestore_to_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Firestore REST Document structure to standard Python dict."""
        if "fields" not in doc:
            return {}
        result = {}
        for key, val_obj in doc["fields"].items():
            result[key] = FirestoreRESTHelper._val_from_firestore(val_obj)
        return result

    @staticmethod
    def _val_from_firestore(val_obj: Dict[str, Any]) -> Any:
        if "nullValue" in val_obj:
            return None
        elif "booleanValue" in val_obj:
            return val_obj["booleanValue"]
        elif "integerValue" in val_obj:
            return int(val_obj["integerValue"])
        elif "doubleValue" in val_obj:
            return float(val_obj["doubleValue"])
        elif "stringValue" in val_obj:
            return val_obj["stringValue"]
        elif "arrayValue" in val_obj:
            values = val_obj["arrayValue"].get("values", [])
            return [FirestoreRESTHelper._val_from_firestore(v) for v in values]
        elif "mapValue" in val_obj:
            sub_fields = val_obj["mapValue"].get("fields", {})
            return {k: FirestoreRESTHelper._val_from_firestore(v) for k, v in sub_fields.items()}
        return None


class DBClient:
    """
    Unified Database and Cloud Storage Client for project `voiceai-6c2e1`.
    
    Primary: `firebase_admin` (if service account configured or initialized).
    Secondary: REST API calls via Google Cloud API Key / Firestore REST endpoint.
    Fallback: Local disk JSON & file cache under `cache/db/` to guarantee zero pipeline breaks.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or default_config
        self.fb_config = self.config.firebase
        self.local_db_dir = self.config.paths.cache_dir / "db"
        self.local_db_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_sdk = False
        self._init_firebase_sdk()

    def _init_firebase_sdk(self) -> None:
        """Initialize firebase_admin if credentials or environment allow."""
        if not _HAS_FIREBASE_ADMIN:
            logger.info("firebase_admin not installed. Defaulting to REST API + Local cache fallback.")
            return

        try:
            if not firebase_admin._apps:
                sa_path = self.fb_config.service_account_path
                if sa_path and os.path.exists(sa_path):
                    cred = credentials.Certificate(sa_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': self.fb_config.storage_bucket,
                        'projectId': self.fb_config.project_id,
                    })
                    self.use_sdk = True
                    logger.info("Firebase Admin SDK initialized with service account certificate.")
                else:
                    # Service account required for firebase_admin on local non-GCP environment
                    self.use_sdk = False
                    logger.info("Service account not provided. Using REST API / Local fallback.")
                    return
            else:
                self.use_sdk = True

            # Verify client access
            if self.use_sdk:
                _ = firestore.client()
        except Exception as err:
            logger.info(f"Firebase SDK active credential test failed ({err}). Using REST API / Local fallback.")
            self.use_sdk = False

    # -------------------------------------------------------------------------
    # Firestore Metadata Persistence
    # -------------------------------------------------------------------------

    def save_voice_profile(self, voice_id: str, version: str, metadata: Dict[str, Any]) -> bool:
        """
        Save voice profile metadata to Firestore collection `voice_profiles`
        under document key `{voice_id}_{version}`.
        """
        doc_id = f"{voice_id}_{version}"
        metadata["voice_id"] = voice_id
        metadata["version"] = version
        metadata["updated_at"] = metadata.get("updated_at", "")

        # Always persist locally first
        self._save_local_doc("voice_profiles", doc_id, metadata)

        # 1. SDK Attempt
        if self.use_sdk:
            try:
                db = firestore.client()
                db.collection("voice_profiles").document(doc_id).set(metadata, merge=True)
                logger.info(f"Saved voice profile {doc_id} via Firebase SDK.")
                return True
            except Exception as e:
                logger.warning(f"Firebase SDK save failed for {doc_id}: {e}. Retrying with REST API.")

        # 2. REST API Attempt
        url = (
            f"https://firestore.googleapis.com/v1/projects/{self.fb_config.project_id}/"
            f"databases/(default)/documents/voice_profiles/{doc_id}"
            f"?key={self.fb_config.api_key}"
        )
        payload = FirestoreRESTHelper.dict_to_firestore(metadata)
        try:
            res = requests.patch(url, json=payload, timeout=5)
            if res.status_code in (200, 201):
                logger.info(f"Saved voice profile {doc_id} via REST API.")
                return True
            else:
                logger.warning(f"Firestore REST error ({res.status_code}): {res.text}")
        except Exception as err:
            logger.warning(f"Firestore REST network failure for {doc_id}: {err}")

        # Local fallback succeeded
        return True

    def get_voice_profile(self, voice_id: str, version: str = "v001") -> Optional[Dict[str, Any]]:
        """Fetch voice profile document for given voice_id and version."""
        doc_id = f"{voice_id}_{version}"

        # 1. SDK Attempt
        if self.use_sdk:
            try:
                db = firestore.client()
                snap = db.collection("voice_profiles").document(doc_id).get()
                if snap.exists:
                    return snap.to_dict()
            except Exception as e:
                logger.warning(f"Firebase SDK fetch failed for {doc_id}: {e}")

        # 2. REST API Attempt
        url = (
            f"https://firestore.googleapis.com/v1/projects/{self.fb_config.project_id}/"
            f"databases/(default)/documents/voice_profiles/{doc_id}"
            f"?key={self.fb_config.api_key}"
        )
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                doc_json = res.json()
                return FirestoreRESTHelper.firestore_to_dict(doc_json)
        except Exception as err:
            logger.warning(f"Firestore REST get failure: {err}")

        # 3. Local fallback
        return self._get_local_doc("voice_profiles", doc_id)

    def save_job_checkpoint(self, job_id: str, state: Dict[str, Any]) -> bool:
        """Save pipeline execution job checkpoint to Firestore collection `job_checkpoints`."""
        state["job_id"] = job_id
        doc_id = job_id
        
        self._save_local_doc("job_checkpoints", doc_id, state)

        if self.use_sdk:
            try:
                db = firestore.client()
                db.collection("job_checkpoints").document(doc_id).set(state, merge=True)
                return True
            except Exception as e:
                logger.warning(f"SDK checkpoint save error: {e}")

        url = (
            f"https://firestore.googleapis.com/v1/projects/{self.fb_config.project_id}/"
            f"databases/(default)/documents/job_checkpoints/{doc_id}"
            f"?key={self.fb_config.api_key}"
        )
        payload = FirestoreRESTHelper.dict_to_firestore(state)
        try:
            res = requests.patch(url, json=payload, timeout=5)
            if res.status_code in (200, 201):
                return True
        except Exception as err:
            logger.warning(f"REST checkpoint save failure: {err}")

        return True

    def get_job_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job checkpoint state."""
        doc_id = job_id
        if self.use_sdk:
            try:
                db = firestore.client()
                snap = db.collection("job_checkpoints").document(doc_id).get()
                if snap.exists:
                    return snap.to_dict()
            except Exception as e:
                logger.warning(f"SDK checkpoint get error: {e}")

        url = (
            f"https://firestore.googleapis.com/v1/projects/{self.fb_config.project_id}/"
            f"databases/(default)/documents/job_checkpoints/{doc_id}"
            f"?key={self.fb_config.api_key}"
        )
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return FirestoreRESTHelper.firestore_to_dict(res.json())
        except Exception as err:
            logger.warning(f"REST checkpoint get failure: {err}")

        return self._get_local_doc("job_checkpoints", doc_id)

    # -------------------------------------------------------------------------
    # Cloud Storage File Persistence
    # -------------------------------------------------------------------------

    def upload_file(self, local_path: Union[str, Path], remote_path: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload local file asset to Firebase Cloud Storage.
        Returns public or Firebase Storage URL.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file to upload does not exist: {local_path}")

        # 1. SDK Attempt
        if self.use_sdk:
            try:
                bucket = storage.bucket(self.fb_config.storage_bucket)
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(str(local_path), content_type=content_type)
                blob.make_public()
                logger.info(f"Uploaded {local_path} to Firebase Storage {remote_path} via SDK.")
                return blob.public_url
            except Exception as e:
                logger.warning(f"Storage SDK upload failed: {e}. Retrying via REST API.")

        # 2. Firebase Storage REST Upload Attempt
        remote_encoded = remote_path.replace("/", "%2F")
        url = (
            f"https://firebasestorage.googleapis.com/v0/b/{self.fb_config.storage_bucket}/o"
            f"?name={remote_encoded}&key={self.fb_config.api_key}"
        )
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            headers = {"Content-Type": content_type}
            res = requests.post(url, data=data, headers=headers, timeout=15)
            if res.status_code in (200, 201):
                download_token = res.json().get("downloadTokens", "")
                public_url = (
                    f"https://firebasestorage.googleapis.com/v0/b/{self.fb_config.storage_bucket}/o/{remote_encoded}"
                    f"?alt=media&token={download_token}"
                )
                logger.info(f"Uploaded {local_path} via REST API: {public_url}")
                return public_url
        except Exception as err:
            logger.warning(f"Firebase Storage REST upload error: {err}")

        # 3. Fallback: simulate storage URL pointing to local path
        return f"file://{local_path.resolve()}"

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Download remote asset from Firebase Cloud Storage to local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_sdk:
            try:
                bucket = storage.bucket(self.fb_config.storage_bucket)
                blob = bucket.blob(remote_path)
                blob.download_to_filename(str(local_path))
                return True
            except Exception as e:
                logger.warning(f"Storage SDK download error: {e}")

        remote_encoded = remote_path.replace("/", "%2F")
        url = (
            f"https://firebasestorage.googleapis.com/v0/b/{self.fb_config.storage_bucket}/o/{remote_encoded}"
            f"?alt=media&key={self.fb_config.api_key}"
        )
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(res.content)
                return True
        except Exception as err:
            logger.warning(f"Firebase Storage REST download error: {err}")

        return False

    # -------------------------------------------------------------------------
    # Local Disk Fallback Helpers
    # -------------------------------------------------------------------------

    def _save_local_doc(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        col_dir = self.local_db_dir / collection
        col_dir.mkdir(parents=True, exist_ok=True)
        doc_file = col_dir / f"{doc_id}.json"
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _get_local_doc(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        doc_file = self.local_db_dir / collection / f"{doc_id}.json"
        if doc_file.exists() and doc_file.stat().st_size > 0:
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading local doc {doc_file}: {e}")
        return None
