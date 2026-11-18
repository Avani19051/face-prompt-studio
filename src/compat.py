"""Small fixes for library-version clashes on hosted notebooks (Kaggle, Colab)."""


def disable_old_torchao_for_peft():
    """Kaggle images ship torchao 0.10, and newer peft raises ImportError for anything
    below 0.16 the moment it looks for it, even though we never use torchao.
    Telling peft that torchao is unavailable sidesteps the check."""
    try:
        import peft.import_utils as iu
        import peft.tuners.lora.torchao as lt

        def _off() -> bool:
            return False

        iu.is_torchao_available = _off
        lt.is_torchao_available = _off
        try:
            import peft.utils.quantization_utils as qu

            qu.is_torchao_available = _off
        except Exception:
            pass
    except Exception:
        pass  # older/newer peft without these modules: nothing to patch
