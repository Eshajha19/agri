import firebase_admin
from firebase_admin import firestore

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

    except Exception:
        return {}

    if not getattr(user_doc, "exists", False):
        return {}

    return dict(user_doc.to_dict() or {})