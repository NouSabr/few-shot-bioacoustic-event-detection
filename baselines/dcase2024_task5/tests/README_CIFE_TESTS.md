# CIFE integration tests

The existing tests are in [test_cife_integration.py](test_cife_integration.py).
Their intended coverage is:

| Test | Intended check |
| --- | --- |
| `test_shape_verification` | Embedding shapes with CIFE enabled and disabled |
| `test_gradient_flow` | Gradients through CIFE and the backbone |
| `test_cife_impact` | Effect of CIFE and its residual scale |
| `test_variable_input_sizes` | Different batch and segment sizes |
| `test_configuration_parameters` | Attention/dropout/residual configuration variants |
| `test_error_handling` | Error handling and the backbone fallback |
| `test_cife_module_standalone` | AudioCIFE independently of the backbone |
| `test_memory_efficiency` | CUDA memory checks when a GPU is available |

From `baselines/dcase2024_task5/`:

```bash
pytest tests/test_cife_integration.py -v
```

This documents test intent, not a passing result. The suite retains legacy
`sys.path`/`models.*` imports; import portability and runtime compatibility have
not been verified by this cleanup. No tests or benchmarks were run, and these
tests are not evidence of compatibility for a particular historical checkpoint.

For current wiring and configuration sources, see the
[CIFE integration guide](../docs/CIFE/README.md).
