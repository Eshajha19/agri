from asyncio.log import logger

import firebase_admin
from firebase_admin import firestore
import logging

logger = logging.getLogger(__name__)
db_firestore = None


def initialize_firebase(logger):
    global db_firestore

    if not firebase_admin._apps:
        try:
            firebase_admin.initialize_app()

            db_firestore = firestore.client()

            logger.info(
                "Firebase Admin: successfully initialized"
            )

        except Exception as e:
            logger.warning(
                "Firebase Admin: could not initialize — "
                "role-gated endpoints will return 503 "
                "until Firestore is reachable. "
                "Reason: %s",
                e,
            )

    return db_firestore


def get_firestore_user_profile(uid: str):
    global db_firestore

    if not db_firestore:
        return {}

    try:
        user_doc = (
            db_firestore
            .collection("users")
            .document(uid)
            .get()
        )
    except Exception as exc:
        logger.error(
        "Firestore profile lookup failed for uid=%s: %s",
        uid,
        exc,
    )
    raise

    if not getattr(user_doc, "exists", False):
        return {}

    return dict(user_doc.to_dict() or {})