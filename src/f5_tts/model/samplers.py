from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Sampler
from tqdm import tqdm


class DynamicBatchSampler(Sampler[list[int]]):
    """Batch by total frame count, with deterministic per-epoch shuffle."""

    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples: int = 0,
        random_seed: int | None = None,
        drop_residual: bool = False,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0
        self.drop_last = True
        self.batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        data_source = self.sampler.data_source
        indexed_lengths = []
        for idx in tqdm(self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"):
            frame_len = data_source.get_frame_len(idx)
            if frame_len is None or frame_len <= 0:
                continue
            indexed_lengths.append((idx, int(frame_len)))

        indexed_lengths.sort(key=lambda elem: elem[1])
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_frames = 0

        for idx, frame_len in tqdm(indexed_lengths, desc=f"Creating dynamic batches with {self.frames_threshold} audio frames per gpu"):
            can_append = batch_frames + frame_len <= self.frames_threshold and (
                self.max_samples == 0 or len(batch) < self.max_samples
            )
            if can_append:
                batch.append(idx)
                batch_frames += frame_len
                continue

            if batch:
                batches.append(batch)
            if frame_len <= self.frames_threshold:
                batch = [idx]
                batch_frames = frame_len
            else:
                batch = []
                batch_frames = 0

        if not self.drop_residual and batch:
            batches.append(batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        # Batches are deterministic (SequentialSampler + sort by length),
        # only their ORDER changes per epoch via __iter__ shuffle.

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            order = torch.randperm(len(self.batches), generator=g).tolist()
            return iter([self.batches[i] for i in order])
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class BucketDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples: int = 64,
        bucket_size: int = 512,
        random_seed: int | None = 42,
        drop_residual: bool = False,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.bucket_size = bucket_size
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0
        self.drop_last = True
        self.batches: list[list[int]] = []
        self._rebuild_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_batches()

    def _rebuild_batches(self):
        data_source = self.sampler.data_source
        indices = list(self.sampler)
        rng = random.Random((self.random_seed or 0) + self.epoch)
        rng.shuffle(indices)
        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        batches: list[list[int]] = []

        for bucket in buckets:
            bucket.sort(key=lambda idx: int(data_source.get_frame_len(idx)))
            batch: list[int] = []
            batch_frames = 0
            for idx in bucket:
                frame_len = int(data_source.get_frame_len(idx))
                if frame_len <= 0:
                    continue
                exceeds_frames = batch and batch_frames + frame_len > self.frames_threshold
                exceeds_samples = self.max_samples and len(batch) >= self.max_samples
                if exceeds_frames or exceeds_samples:
                    batches.append(batch)
                    batch = []
                    batch_frames = 0
                if frame_len <= self.frames_threshold:
                    batch.append(idx)
                    batch_frames += frame_len
            if batch and not self.drop_residual:
                batches.append(batch)

        rng.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)



class SpeakerAwareBucketDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples=64,
        bucket_size: int = 512,
        max_speakers_per_batch: int = 8,
        max_samples_per_speaker: int = 8,
        random_seed=None,
        drop_residual: bool = False,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.bucket_size = bucket_size
        self.max_speakers_per_batch = max_speakers_per_batch
        self.max_samples_per_speaker = max_samples_per_speaker
        self.random_seed = random_seed
        self.epoch = 0
        self.drop_residual = drop_residual
        self.drop_last = True
        self.batches = []
        self._rebuild_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_batches()

    def _shuffle_indices(self, indices):
        if self.random_seed is not None:
            rng = random.Random(self.random_seed + self.epoch)
            rng.shuffle(indices)
        else:
            random.shuffle(indices)
        return indices

    def _shuffle_batches(self, batches):
        if self.random_seed is not None:
            rng = random.Random(self.random_seed + self.epoch + 99991)
            rng.shuffle(batches)
        else:
            random.shuffle(batches)
        return batches

    def _can_fit(self, frame_len, batch_frames, batch_size, batch_max_len=0):
        if frame_len is None or frame_len <= 0:
            return False
        if frame_len > self.frames_threshold:
            return False
        if self.max_samples != 0 and batch_size >= self.max_samples:
            return False
        if batch_size > 0 and batch_frames + frame_len > self.frames_threshold:
            return False
        # OOM guard: limit padding waste. Real memory usage = max_len × batch_size.
        # If new sample is much longer than current max, total padded frames explode.
        new_max = max(batch_max_len, frame_len)
        if batch_size > 0 and new_max * (batch_size + 1) > self.frames_threshold:
            return False
        return True

    def _try_fill_batch_from_candidates(
        self,
        candidates,
        data_source,
        batch,
        batch_frames,
        speaker_counter,
        prefer_existing_speakers,
    ):
        remaining = []
        batch_max_len = max((data_source.get_frame_len(i) for i in batch), default=0)
        for idx in candidates:
            frame_len = data_source.get_frame_len(idx)
            speaker = data_source.get_speaker(idx)

            if frame_len is None or frame_len <= 0:
                continue

            if prefer_existing_speakers and len(batch) > 0 and speaker not in speaker_counter:
                remaining.append(idx)
                continue

            if not self._can_fit(frame_len, batch_frames, len(batch), batch_max_len):
                remaining.append(idx)
                continue

            is_new_speaker = speaker not in speaker_counter
            too_many_speakers = is_new_speaker and len(speaker_counter) >= self.max_speakers_per_batch
            too_many_for_speaker = speaker_counter[speaker] >= self.max_samples_per_speaker

            if too_many_speakers or too_many_for_speaker:
                remaining.append(idx)
                continue

            batch.append(idx)
            batch_frames += frame_len
            batch_max_len = max(batch_max_len, frame_len)
            speaker_counter[speaker] += 1

        return batch, batch_frames, speaker_counter, remaining

    def _rebuild_batches(self):
        data_source = self.sampler.data_source
        indices = list(self.sampler)
        indices = self._shuffle_indices(indices)

        buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        batches = []

        for bucket in buckets:
            bucket = sorted(bucket, key=lambda idx: data_source.get_frame_len(idx))
            pending = bucket[:]

            while pending:
                batch = []
                batch_frames = 0
                speaker_counter = Counter()

                seed_idx = pending.pop(0)
                seed_frame_len = data_source.get_frame_len(seed_idx)
                seed_speaker = data_source.get_speaker(seed_idx)

                if seed_frame_len is None or seed_frame_len <= 0 or seed_frame_len > self.frames_threshold:
                    continue

                batch.append(seed_idx)
                batch_frames = seed_frame_len
                speaker_counter[seed_speaker] = 1

                batch, batch_frames, speaker_counter, pending = self._try_fill_batch_from_candidates(
                    pending,
                    data_source,
                    batch,
                    batch_frames,
                    speaker_counter,
                    prefer_existing_speakers=True,
                )

                batch, batch_frames, speaker_counter, pending = self._try_fill_batch_from_candidates(
                    pending,
                    data_source,
                    batch,
                    batch_frames,
                    speaker_counter,
                    prefer_existing_speakers=False,
                )

                if batch:
                    batches.append(batch)

        if self.drop_residual:
            filtered_batches = []
            for batch in batches:
                if self.max_samples > 0:
                    if len(batch) == self.max_samples:
                        filtered_batches.append(batch)
                else:
                    filtered_batches.append(batch)
            batches = filtered_batches

        self.batches = self._shuffle_batches(batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



class SpeakerBalancedDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples=64,
        speakers_per_batch: int = 8,
        samples_per_speaker: int = 4,
        random_seed=None,
        drop_residual: bool = False,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.random_seed = random_seed
        self.epoch = 0
        self.drop_residual = drop_residual
        self.drop_last = True
        self.batches = []
        self._rebuild_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_batches()

    def _rebuild_batches(self):
        data_source = self.sampler.data_source
        indices = list(self.sampler)

        if self.random_seed is not None:
            rng = random.Random(self.random_seed + self.epoch)
            rng.shuffle(indices)
        else:
            rng = random
            rng.shuffle(indices)

        speaker_to_indices = defaultdict(list)
        for idx in indices:
            fl = data_source.get_frame_len(idx)
            if fl is None or fl <= 0 or fl > self.frames_threshold:
                continue
            sp = data_source.get_speaker(idx)
            speaker_to_indices[sp].append(idx)

        for sp in speaker_to_indices:
            speaker_to_indices[sp].sort(key=lambda i: data_source.get_frame_len(i))

        active_speakers = [sp for sp, items in speaker_to_indices.items() if items]
        rng.shuffle(active_speakers)

        batches = []

        while active_speakers:
            chosen_speakers = []
            for sp in active_speakers:
                if speaker_to_indices[sp]:
                    chosen_speakers.append(sp)
                if len(chosen_speakers) >= self.speakers_per_batch:
                    break

            if len(chosen_speakers) < min(self.speakers_per_batch, len(active_speakers)) and self.drop_residual:
                break
            if not chosen_speakers:
                break

            batch = []
            frames = 0
            kept_speakers = 0

            for sp in chosen_speakers:
                taken = 0
                remaining = []
                local_added = []
                local_frames = 0

                for idx in speaker_to_indices[sp]:
                    fl = data_source.get_frame_len(idx)

                    if taken >= self.samples_per_speaker:
                        remaining.append(idx)
                        continue

                    if len(batch) + len(local_added) >= self.max_samples:
                        remaining.append(idx)
                        continue

                    if (batch or local_added) and frames + local_frames + fl > self.frames_threshold:
                        remaining.append(idx)
                        continue

                    local_added.append(idx)
                    local_frames += fl
                    taken += 1

                if taken == self.samples_per_speaker:
                    batch.extend(local_added)
                    frames += local_frames
                    kept_speakers += 1
                    # return unconsumed plus any trailing items
                    consumed = set(local_added)
                    speaker_to_indices[sp] = [i for i in speaker_to_indices[sp] if i not in consumed]
                else:
                    # could not satisfy full quota for this speaker in this batch
                    speaker_to_indices[sp] = local_added + remaining + [i for i in speaker_to_indices[sp] if i not in local_added and i not in remaining]

            if kept_speakers > 0 and (not self.drop_residual or kept_speakers == len(chosen_speakers)):
                batch.sort(key=lambda i: data_source.get_frame_len(i))
                batches.append(batch)

            active_speakers = [sp for sp in active_speakers if speaker_to_indices[sp]]
            rng.shuffle(active_speakers)

        if self.random_seed is not None:
            rng2 = random.Random(self.random_seed + self.epoch + 99991)
            rng2.shuffle(batches)
        else:
            random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)
