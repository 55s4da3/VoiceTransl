"""Compatibility facade for the speech-recognition bridges.

The Qwen UI historically imported helpers from this module.  The actual
validation and subprocess behavior now live in the dedicated bridge modules.
"""

import crispasr_bridge


def _list_crispasr_models():
    return crispasr_bridge.list_models()


def _list_crispasr_aligners():
    return crispasr_bridge.list_aligners()


def _list_crispasr_backends():
    try:
        return crispasr_bridge.list_backends()
    except crispasr_bridge.CrispASRBackendDiscoveryError:
        return list(crispasr_bridge.CRISPASR_BACKEND_FALLBACK)


def _build_crispasr_command(
    input_file,
    output_file,
    model_file,
    language,
    param_crispasr,
    aligner_file=None,
    backend=None,
):
    return crispasr_bridge._build_crispasr_command(
        input_file,
        output_file,
        model_file,
        language,
        param_crispasr,
        aligner_file=aligner_file,
        backend=backend,
    )
