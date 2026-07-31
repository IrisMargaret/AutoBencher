import smoke_test

from verify_project import verify_project


def test_offline_project_health_check_passes():
    report = verify_project()
    assert report["status"] == "passed", report
    assert report["checks"]["python_syntax"]["bytecode_written"] is False
    assert report["checks"]["repository_hygiene"]["literal_credentials_found"] == 0


def test_offline_smoke_never_initializes_api(monkeypatch, capsys):
    def fail_api(_model_name):
        raise AssertionError("offline smoke attempted to initialize an API")

    monkeypatch.setattr(smoke_test, "process_args_for_models", fail_api)
    assert smoke_test.main([]) == 0
    assert "no API request was sent" in capsys.readouterr().out
