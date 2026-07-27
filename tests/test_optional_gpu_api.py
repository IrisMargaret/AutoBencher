import os

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("AUTOBENCHER_RUN_GPU_TESTS") != "1",
    reason="Set AUTOBENCHER_RUN_GPU_TESTS=1 on a configured CUDA host.",
)
def test_gpu_environment_opt_in():
    import torch

    assert torch.cuda.is_available()


@pytest.mark.api
@pytest.mark.skipif(
    os.getenv("AUTOBENCHER_RUN_API_TESTS") != "1",
    reason="Set AUTOBENCHER_RUN_API_TESTS=1 with API credentials.",
)
def test_api_environment_opt_in():
    assert os.getenv("DEEPSEEK_API_KEY") or os.getenv("ARK_API_KEY")

