"""
STT Model Singleton Manager

Thread-safe singleton manager for FasterWhisper model.
Keeps the model loaded in memory and provides resource release mechanism
for training/inference operations.

Usage:
    # Get model instance (loads if not already loaded)
    manager = STTModelManager.get_instance()
    model = manager.acquire_model(model_path, device, precision)

    # Use model...
    segments, info = model.transcribe(audio_path, ...)

    # Release model for training/inference (frees GPU memory)
    manager.release_model()

    # Context manager usage
    with manager.get_model(model_path, device, precision) as model:
        segments, info = model.transcribe(audio_path, ...)
"""
import threading
import logging
from typing import Optional
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class STTModelManager:
    """
    Singleton manager for FasterWhisper STT model.

    Thread-safe implementation that keeps model loaded in memory
    and allows explicit resource release for training/inference operations.
    """

    _instance: Optional['STTModelManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Private constructor. Use get_instance() instead."""
        if STTModelManager._instance is not None:
            raise RuntimeError("Use get_instance() to get singleton instance")

        self._model: Optional[object] = None  # WhisperModel instance
        self._model_path: Optional[str] = None
        self._device: Optional[str] = None
        self._precision: Optional[str] = None
        self._model_lock = threading.RLock()
        self._ref_count = 0  # Track number of active users

        logger.info("STTModelManager singleton initialized")

    @classmethod
    def get_instance(cls) -> 'STTModelManager':
        """
        Get singleton instance of STTModelManager.

        Thread-safe lazy initialization.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def acquire_model(
        self,
        model_path: str,
        device: str = "cuda",
        precision: str = "float16"
    ) -> object:
        """
        Acquire STT model instance.

        If model is already loaded with same parameters, returns cached instance.
        Otherwise, loads new model (releasing old one if exists).

        Args:
            model_path: Path to FasterWhisper model
            device: Device to load model on ("cuda" or "cpu")
            precision: Model precision ("float16", "float32", or "int8")

        Returns:
            WhisperModel instance
        """
        with self._model_lock:
            self._ref_count += 1

            # Check if we need to reload model
            needs_reload = (
                self._model is None or
                self._model_path != model_path or
                self._device != device or
                self._precision != precision
            )

            if needs_reload:
                # Release old model if exists
                if self._model is not None:
                    logger.info(
                        f"Releasing old STT model (path={self._model_path}, "
                        f"device={self._device}, precision={self._precision})"
                    )
                    self._release_internal()

                # Load new model
                logger.info(
                    f"Loading STT model (path={model_path}, "
                    f"device={device}, precision={precision})"
                )

                try:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(
                        model_path,
                        device=device,
                        compute_type=precision
                    )
                    self._model_path = model_path
                    self._device = device
                    self._precision = precision

                    logger.info(f"STT model loaded successfully (ref_count={self._ref_count})")

                except Exception as e:
                    logger.error(f"Failed to load STT model: {e}", exc_info=True)
                    self._ref_count -= 1
                    raise
            else:
                logger.debug(
                    f"Reusing cached STT model (ref_count={self._ref_count}, "
                    f"path={model_path})"
                )

            return self._model

    def release_reference(self):
        """
        Release one reference to the model.

        Model stays in memory even when ref_count reaches 0.
        Use release_model() to explicitly free GPU memory.
        """
        with self._model_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                logger.debug(f"Released STT model reference (ref_count={self._ref_count})")

    def release_model(self, force: bool = False):
        """
        Release model from memory.

        Frees GPU/CPU memory used by the model.
        Useful before starting training or inference operations.

        Args:
            force: If True, releases model even if ref_count > 0
        """
        with self._model_lock:
            if self._model is None:
                logger.debug("STT model already released")
                return

            if not force and self._ref_count > 0:
                logger.warning(
                    f"Cannot release STT model: {self._ref_count} active references. "
                    f"Use force=True to override."
                )
                return

            logger.info(
                f"Releasing STT model (path={self._model_path}, "
                f"device={self._device}, ref_count={self._ref_count})"
            )

            self._release_internal()
            self._ref_count = 0

    def _release_internal(self):
        """Internal method to release model resources."""
        if self._model is not None:
            # Delete model instance
            del self._model
            self._model = None
            self._model_path = None
            self._device = None
            self._precision = None

            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after STT model release")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        with self._model_lock:
            return self._model is not None

    def get_model_info(self) -> dict:
        """
        Get information about currently loaded model.

        Returns:
            Dict with model_path, device, precision, ref_count, is_loaded
        """
        with self._model_lock:
            return {
                "is_loaded": self._model is not None,
                "model_path": self._model_path,
                "device": self._device,
                "precision": self._precision,
                "ref_count": self._ref_count
            }

    @contextmanager
    def get_model(
        self,
        model_path: str,
        device: str = "cuda",
        precision: str = "float16"
    ):
        """
        Context manager for acquiring and releasing model.

        Usage:
            manager = STTModelManager.get_instance()
            with manager.get_model(model_path, device, precision) as model:
                segments, info = model.transcribe(audio_path, ...)

        Args:
            model_path: Path to FasterWhisper model
            device: Device to load model on
            precision: Model precision

        Yields:
            WhisperModel instance
        """
        model = self.acquire_model(model_path, device, precision)
        try:
            yield model
        finally:
            self.release_reference()


# Global convenience function
def get_stt_model_manager() -> STTModelManager:
    """Get singleton STT model manager instance."""
    return STTModelManager.get_instance()
