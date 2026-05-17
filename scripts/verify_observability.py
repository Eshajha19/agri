import importlib.util, os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
main_path = os.path.join(repo_root, 'main.py')
try:
    spec = importlib.util.spec_from_file_location('_main_for_verify', main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = getattr(mod, 'app', None)
    if app is None:
        print('ERROR: app not found in main.py')
        sys.exit(3)
    # Check metrics endpoint presence
    metrics_ok = any(getattr(r, 'path', '') == '/metrics' for r in app.routes)
    # Check tracing: try to import opentelemetry and check tracer provider
    try:
        from opentelemetry import trace
        tp = trace.get_tracer_provider()
        tracing_ok = tp is not None
    except Exception:
        tracing_ok = False
    print('metrics:', metrics_ok)
    print('tracing:', tracing_ok)
    if not metrics_ok or not tracing_ok:
        print('OBSERVABILITY CHECK FAILED')
        sys.exit(2)
    print('OBSERVABILITY CHECK PASSED')
    sys.exit(0)
except Exception as e:
    print('IMPORT ERROR:', e)
    sys.exit(4)
