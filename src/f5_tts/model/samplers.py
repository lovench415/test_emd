from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Sampler
from tqdm import tqdm


# Distinct prime offsets so the per-(seed, epoch, round) reshuffle keys of the
# different samplers don't collide and so each reshuffle round differs. Named here
# once instead of being copied as magic numbers across every sampler.
_EPOCH_STRIDE = 100003
_ROUND_SALT = 99991


class _DynamicBatchSamplerBase(Sampler[list[int]]):
    """Shared behaviour for the dynamic batch samplers.

    Centralizes the two pieces that were previously copy-pasted into every sampler
    (and thus drifted when one was fixed but the others weren't):

      • _reshuffle_iter: iterate self.batches, and every rebuild_every_n_steps yield
        a SEEDED reshuffle of the remainder keyed by (seed, epoch, round). Seeding
        is what keeps DDP ranks in lockstep (a bare random.shuffle reads the global
        RNG, which diverges across processes); the round counter keeps successive
        reshuffles distinct yet reproducible. Falls back to the global RNG only when
        no seed was set.

      • __len__: a STABLE nominal length cached on first access. Subclasses that
        repack per epoch (set_epoch → _rebuild_batches) otherwise return a count that
        drifts a few % each epoch, which desyncs a step-based LR scheduler.

    Subclasses set self.batches, self.random_seed, self.epoch, and
    self.rebuild_every_n_steps, then use these helpers.

    ─────────────────────────────────────────────────────────────────────────────
    SHARDING MODEL (read before changing anything DDP-related)

    These are BATCH samplers that do NOT shard by rank — there is no rank /
    num_replicas here. Each builds ONE GLOBAL list of batches, and the per-rank
    split is done by Accelerate ON TOP of that list (accelerator.prepare wraps the
    DataLoader and hands out batches round-robin across processes). That choice is
    what makes the schedule-horizon math elsewhere use
    `len(dataloader) / num_processes` rather than len directly.

    This pattern is CORRECT, but only while three conditions hold — break any one
    and ranks silently desync (process different data, or hang on all-reduce):

      1. DETERMINISTIC GLOBAL LIST. Every rank must build the IDENTICAL global
         batch list, or round-robin hands different batches to each rank. The plain
         DynamicBatchSampler is deterministic (it sorts by length). The bucket /
         speaker samplers shuffle, so they are deterministic ONLY when random_seed
         is not None. The trainer passes --seed (default 666), so this holds; if a
         caller ever sets seed=None on multi-GPU, the global lists diverge.

      2. EVEN BATCH COUNT. If the number of global batches isn't divisible by
         num_processes, round-robin gives one rank an extra batch. DDP requires the
         SAME number of optimizer steps per rank or it hangs on gradient all-reduce.
         Accelerate's even_batches (default True) handles this by trimming/padding;
         don't disable it for these samplers without adding explicit drop_last logic.

      3. SYNCHRONIZED set_epoch. Every rank must call set_epoch(epoch) with the same
         epoch so the per-epoch repack (same seed+epoch) reproduces the same global
         list on every rank. The training loop does this once per epoch for all ranks.

    If any of these becomes hard to guarantee, the robust alternative is a
    self-sharding sampler (own rank/num_replicas that slices its own shard), which
    removes the dependency on Accelerate's round-robin entirely.
    """

    def _reshuffle_iter(self, salt: int = 0, batches=None):
        remaining = list(self.batches if batches is None else batches)
        step = 0
        reshuffle_round = 0
        while remaining:
            yield remaining.pop(0)
            step += 1
            if (self.rebuild_every_n_steps > 0
                    and step % self.rebuild_every_n_steps == 0 and remaining):
                if self.random_seed is not None:
                    r = random.Random(self.random_seed
                                      + self.epoch * _EPOCH_STRIDE
                                      + salt + reshuffle_round)
                    r.shuffle(remaining)
                else:
                    random.shuffle(remaining)
                reshuffle_round += 1

    def _stable_len(self) -> int:
        if getattr(self, "_nominal_len", None) is None:
            self._nominal_len = len(self.batches)
        return self._nominal_len


class DynamicBatchSampler(_DynamicBatchSamplerBase):
    """Batch by total frame count, with deterministic per-epoch shuffle."""

    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples: int = 0,
        random_seed: int | None = None,
        drop_residual: bool = False,
        rebuild_every_n_steps: int = 0,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0
        self.drop_last = True
        self.rebuild_every_n_steps = rebuild_every_n_steps
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
            batches = [self.batches[i] for i in order]
        else:
            batches = list(self.batches)

        if self.rebuild_every_n_steps <= 0:
            return iter(batches)
        return self._reshuffle_iter(batches=batches)

    def __len__(self):
        return len(self.batches)


class BucketDynamicBatchSampler(_DynamicBatchSamplerBase):
    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples: int = 64,
        bucket_size: int = 512,
        random_seed: int | None = 42,
        drop_residual: bool = False,
        rebuild_every_n_steps: int = 0,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.bucket_size = bucket_size
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0
        self.drop_last = True
        self.rebuild_every_n_steps = rebuild_every_n_steps
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
        if self.rebuild_every_n_steps <= 0:
            return iter(self.batches)
        return self._reshuffle_iter()

    def __len__(self):
        return self._stable_len()



class SpeakerAwareBucketDynamicBatchSampler(_DynamicBatchSamplerBase):
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
        rebuild_every_n_steps: int = 0,
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
        self.rebuild_every_n_steps = rebuild_every_n_steps  # 0 = only at epoch start
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
        get_fl,
        get_sp,
        batch,
        batch_frames,
        batch_max_len,
        speaker_counter,
        prefer_existing_speakers,
    ):
        remaining = []
        for idx in candidates:
            frame_len = get_fl(idx)
            speaker = get_sp(idx)

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

        return batch, batch_frames, batch_max_len, speaker_counter, remaining

    def _rebuild_batches(self):
        data_source = self.sampler.data_source
        indices = list(self.sampler)
        indices = self._shuffle_indices(indices)

        # Cache frame_len and speaker for all indices once — avoids O(N²) lookups.
        # get_frame_len/get_speaker may touch HF dataset rows which are slow.
        frame_len_cache = {}
        speaker_cache = {}
        for idx in indices:
            fl = data_source.get_frame_len(idx)
            frame_len_cache[idx] = fl
            if fl and fl > 0:
                speaker_cache[idx] = data_source.get_speaker(idx)

        def get_fl(idx):
            return frame_len_cache.get(idx)

        def get_sp(idx):
            return speaker_cache.get(idx, "default")

        buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        batches = []

        for bucket in buckets:
            bucket = sorted(bucket, key=lambda idx: frame_len_cache.get(idx) or 0)
            pending = bucket[:]

            while pending:
                batch = []
                batch_frames = 0
                batch_max_len = 0
                speaker_counter = Counter()

                seed_idx = pending.pop(0)
                seed_frame_len = get_fl(seed_idx)
                seed_speaker = get_sp(seed_idx)

                if seed_frame_len is None or seed_frame_len <= 0 or seed_frame_len > self.frames_threshold:
                    continue

                batch.append(seed_idx)
                batch_frames = seed_frame_len
                batch_max_len = seed_frame_len
                speaker_counter[seed_speaker] = 1

                batch, batch_frames, batch_max_len, speaker_counter, pending = \
                    self._try_fill_batch_from_candidates(
                        pending, get_fl, get_sp,
                        batch, batch_frames, batch_max_len,
                        speaker_counter, prefer_existing_speakers=True,
                    )

                batch, batch_frames, batch_max_len, speaker_counter, pending = \
                    self._try_fill_batch_from_candidates(
                        pending, get_fl, get_sp,
                        batch, batch_frames, batch_max_len,
                        speaker_counter, prefer_existing_speakers=False,
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
        if self.rebuild_every_n_steps <= 0:
            return iter(self.batches)
        # salt=_ROUND_SALT keeps this sampler's reshuffle keys distinct from the
        # plain dynamic sampler's, so the two don't produce correlated orders.
        return self._reshuffle_iter(salt=_ROUND_SALT)

    def __len__(self):
        return self._stable_len()



class SpeakerBalancedDynamicBatchSampler(_DynamicBatchSamplerBase):
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

                    if self.max_samples and len(batch) + len(local_added) >= self.max_samples:
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

            # Guarantee progress. A speaker with fewer than samples_per_speaker items
            # left can never fill its quota: its items get returned unchanged each
            # round and it stays "active" forever → infinite loop. If this round
            # produced no batch (kept_speakers == 0), drop every speaker that cannot
            # reach the quota so their leftovers become residual instead of spinning.
            # When progress was made, only prune now-empty speakers as before.
            if kept_speakers == 0:
                before = len(active_speakers)
                active_speakers = [
                    sp for sp in active_speakers
                    if len(speaker_to_indices[sp]) >= self.samples_per_speaker
                ]
                # Absolute safety net: if we still made no progress (same active set),
                # stop rather than risk spinning on an unforeseen edge case.
                if len(active_speakers) == before:
                    break
            else:
                active_speakers = [sp for sp in active_speakers if speaker_to_indices[sp]]
            rng.shuffle(active_speakers)

        if self.random_seed is not None:
            rng2 = random.Random(self.random_seed + self.epoch + 99991)
            rng2.shuffle(batches)
        else:
            random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        # This sampler re-packs fully on set_epoch and does not reshuffle mid-epoch,
        # so it just yields the current batches.
        for b in self.batches:
            yield b

    def __len__(self):
        return self._stable_len()
