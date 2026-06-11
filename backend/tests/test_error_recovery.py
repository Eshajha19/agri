from backend.error_recovery_middleware import looks_like_binary, looks_like_base64

def test_detect_gzip_magic():
    assert looks_like_binary(b"\x1F\x8B...rest") is True

def test_detect_zip_magic():
    assert looks_like_binary(b"PK\x03\x04...rest") is True

def test_detect_base64():
    sample = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="  # base64 for ABC...
    assert looks_like_base64(sample) is True
