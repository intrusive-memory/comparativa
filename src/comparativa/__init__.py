"""comparativa — all-Python Fountain-to-speech reference pipeline.

Parses Fountain screenplays and generates episode audio with MLX speech models
so the resulting output can be compared, under controlled conditions, against
the existing Swift stack (Produciesta / SwiftVoxAlta).
"""

__version__ = "0.1.0"

#: The mlx-audio release this pipeline is pinned to and validated against.
MLX_AUDIO_VERSION = "0.4.8"

__all__ = ["__version__", "MLX_AUDIO_VERSION"]
