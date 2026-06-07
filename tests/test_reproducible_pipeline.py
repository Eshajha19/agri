import os
import tempfile

from ml.ci_pipeline import validate_csv_schema, sign_file_hmac, verify_file_hmac


def test_validate_train_csv_exists_and_headers():
    path = "Train.csv"
    assert os.path.exists(path), "Train.csv must exist for this test"
    # basic validation: at least 3 columns
    assert validate_csv_schema(path, min_columns=3) is True


def test_sign_and_verify_temp_file():
    key = "test-signing-key"
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"small test payload")
        tf.flush()
        p = tf.name

    sig = sign_file_hmac(p, key)
    assert isinstance(sig, str) and len(sig) == 64
    assert verify_file_hmac(p, sig, key) is True

    os.remove(p)
