import traceback

print('START')
try:
    import app
    print('IMPORT_APP_OK')
except Exception:
    print('IMPORT_APP_FAILED')
    traceback.print_exc()

try:
    from src.pipeline import answer_question
    print('IMPORT_PIPELINE_OK')
except Exception:
    print('IMPORT_PIPELINE_FAILED')
    traceback.print_exc()
