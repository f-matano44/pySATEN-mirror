import copy
import wave

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from torch.utils.data import DataLoader


class MarbleNet:
    def __init__(self):
        self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            "vad_marblenet"
        )
        self.cfg = copy.deepcopy(self.vad_model._cfg)
        self.vad_model.preprocessor = self.vad_model.from_config_dict(
            self.cfg.preprocessor
        )
        self.vad_model.eval()
        self.vad_model = self.vad_model.to(self.vad_model.device)

    # class for streaming frame-based VAD
    # 1) use reset() method to reset FrameVAD's state
    # 2) call transcribe(frame) to do VAD on
    #    contiguous signal's frames
    # To simplify the flow, we use single threshold to binarize predictions.
    class FrameVAD:
        def __init__(
            self,
            vad_model,
            cfg,
            model_definition,
            threshold=0.5,
            frame_len=2,
            frame_overlap=2.5,
            offset=10,
        ):
            self.vad_model = vad_model
            self.cfg = cfg
            """
            Args:
            threshold: If prob of speech is larger than threshold,
                        classify the segment to be speech.
            frame_len: frame's duration, seconds
            frame_overlap: duration of overlaps before and
                            after current frame, seconds
            offset: number of symbols to drop for smooth streaming
            """
            self.vocab = list(model_definition["labels"])
            self.vocab.append("_")

            self.sr = model_definition["sample_rate"]
            self.threshold = threshold
            self.frame_len = frame_len
            self.n_frame_len = int(frame_len * self.sr)
            self.frame_overlap = frame_overlap
            self.n_frame_overlap = int(frame_overlap * self.sr)
            timestep_duration = model_definition["AudioToMFCCPreprocessor"][
                "window_stride"
            ]
            for block in model_definition["JasperEncoder"]["jasper"]:
                timestep_duration *= block["stride"][0] ** block["repeat"]
            self.buffer = np.zeros(
                shape=2 * self.n_frame_overlap + self.n_frame_len, dtype=np.float32
            )
            self.offset = offset
            self.reset()

        # simple data layer to pass audio signal
        class AudioDataLayer(IterableDataset):
            @property
            def output_types(self):
                return {
                    "audio_signal": NeuralType(
                        ("B", "T"), AudioSignal(freq=self._sample_rate)
                    ),
                    "a_sig_length": NeuralType(tuple("B"), LengthsType()),
                }

            def __init__(self, sample_rate):
                super().__init__()
                self._sample_rate = sample_rate
                self.output = True

            def __iter__(self):
                return self

            def __next__(self):
                if not self.output:
                    raise StopIteration
                self.output = False
                return torch.as_tensor(self.signal, dtype=torch.float32), torch.as_tensor(
                    self.signal_shape, dtype=torch.int64
                )

            def set_signal(self, signal):
                self.signal = signal.astype(np.float32) / 32768.0
                self.signal_shape = self.signal.size
                self.output = True

            def __len__(self):
                return 1

        def _decode(self, frame, offset=0):
            assert len(frame) == self.n_frame_len
            self.buffer[: -self.n_frame_len] = self.buffer[self.n_frame_len :]
            self.buffer[-self.n_frame_len :] = frame
            logits = self.infer_signal(self.vad_model, self.buffer).cpu().numpy()[0]
            decoded = self._greedy_decoder(self.threshold, logits, self.vocab)
            return decoded

        @torch.no_grad()
        def transcribe(self, frame=None):
            if frame is None:
                frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
            if len(frame) < self.n_frame_len:
                frame = np.pad(frame, [0, self.n_frame_len - len(frame)], "constant")
            unmerged = self._decode(frame, self.offset)
            return unmerged

        def reset(self):
            """
            Reset frame_history and decoder's state
            """
            self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
            self.prev_char = ""

        @staticmethod
        def _greedy_decoder(threshold, logits, vocab):
            s = []
            if logits.shape[0]:
                probs = torch.softmax(torch.as_tensor(logits), dim=-1)
                probas, _ = torch.max(probs, dim=-1)
                probas_s = probs[1].item()
                preds = 1 if probas_s >= threshold else 0
                s = [
                    preds,
                    str(vocab[preds]),
                    probs[0].item(),
                    probs[1].item(),
                    str(logits),
                ]
            return s

        # inference method for audio signal (single instance)
        def infer_signal(self, model, signal):
            data_layer = self.AudioDataLayer(sample_rate=self.cfg.train_ds.sample_rate)
            data_loader = DataLoader(
                data_layer, batch_size=1, collate_fn=data_layer.collate_fn
            )
            data_layer.set_signal(signal)
            batch = next(iter(data_loader))
            audio_signal, audio_signal_len = batch
            audio_signal, audio_signal_len = audio_signal.to(
                self.vad_model.device
            ), audio_signal_len.to(self.vad_model.device)
            logits = model.forward(
                input_signal=audio_signal, input_signal_length=audio_signal_len
            )
            return logits

    def offline_inference(self, wave_file, STEP=0.025, WINDOW_SIZE=0.5, threshold=0.5):
        FRAME_LEN = STEP  # infer every STEP seconds
        SAMPLE_RATE = 16000  # sample rate, Hz

        CHUNK_SIZE = int(FRAME_LEN * SAMPLE_RATE)

        vad = MarbleNet.FrameVAD(
            self.vad_model,
            self.cfg,
            model_definition={
                "sample_rate": SAMPLE_RATE,
                "AudioToMFCCPreprocessor": self.cfg.preprocessor,
                "JasperEncoder": self.cfg.encoder,
                "labels": self.cfg.labels,
            },
            threshold=threshold,
            frame_len=FRAME_LEN,
            frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2,
            offset=0,
        )

        preds = []

        wf = wave.open(wave_file, "rb")
        data = wf.readframes(CHUNK_SIZE)

        while len(data) > 0:
            data = wf.readframes(CHUNK_SIZE)
            signal = np.frombuffer(data, dtype=np.int16)
            result = vad.transcribe(signal)
            preds.append(result[0])

        vad.reset()
        preds = np.array(preds)

        start = np.min(np.where(preds > 0)) * STEP if np.any(preds > 0) else 0
        end = np.max(np.where(preds > 0)) * STEP if np.any(preds > 0) else 0

        return start, end

    def vad_test(self, x, fs):
        x = librosa.resample(x, orig_sr=fs, target_sr=16000)
        sf.write("temp.wav", x, 16000)

        step = 0.025
        return self.offline_inference("temp.wav", step)
