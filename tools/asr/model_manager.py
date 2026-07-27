"""Thread-safe singleton manager for the FasterWhisper STT model: keeps the model loaded in memory, reuses it when parameters match, and provides explicit resource release (and a context manager) to free GPU memory for training/inference."""

import threading
import logging
from typing import Optional
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class STTModelManager:
    _instance: Optional['STTModelManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        if STTModelManager._instance is not None:
            raise RuntimeError("Use get_instance() to get singleton instance")

        self._model: Optional[object] = None
        self._model_path: Optional[str] = None
        self._device: Optional[str] = None
        self._precision: Optional[str] = None
        self._model_lock = threading.RLock()
        self._ref_count = 0

        logger.info("STTModelManager singleton initialized")

    @classmethod
    def get_instance(cls) -> 'STTModelManager':
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
        with self._model_lock:
            self._ref_count += 1

            needs_reload = (
                self._model is None or
                self._model_path != model_path or
                self._device != device or
                self._precision != precision
            )

            if needs_reload:
                if self._model is not None:
                    logger.info(
                        f"Releasing old STT model (path={self._model_path}, "
                        f"device={self._device}, precision={self._precision})"
                    )
                    self._release_internal()

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
        with self._model_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                logger.debug(f"Released STT model reference (ref_count={self._ref_count})")

    def release_model(self, force: bool = False):
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
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = None
            self._device = None
            self._precision = None

            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after STT model release")

    def is_loaded(self) -> bool:
        with self._model_lock:
            return self._model is not None

    def get_model_info(self) -> dict:
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
        model = self.acquire_model(model_path, device, precision)
        try:
            yield model
        finally:
            self.release_reference()


def get_stt_model_manager() -> STTModelManager:
    return STTModelManager.get_instance()
