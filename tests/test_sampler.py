"""CPU regressions for complete distributed sampling and resumed epoch order."""

from collections import Counter
import math
import unittest

import torch
from swift.dataloader import DataLoaderShard

from jag.trainer import GroundingBatchSampler


def batches(size, batch_size, rank, *, world_size=2, epoch=0, skip=0):
    sampler = GroundingBatchSampler(
        size, batch_size, num_replicas=world_size, rank=rank,
        seed=42, shuffle=True, skip_batches=skip,
    )
    sampler.set_epoch(epoch)
    return sampler, list(sampler)


class GroundingSamplerTests(unittest.TestCase):
    def test_every_original_row_is_seen_and_ranks_have_equal_work(self):
        # Includes fewer samples than ranks, odd sizes, and partial microbatches.
        for world_size in (2, 4):
            for size in (1, 2, 3, 7, 9, 10, 13):
                for batch_size in (1, 2, 4):
                    for epoch in (0, 1, 3):
                        with self.subTest(world_size=world_size, size=size,
                                          batch_size=batch_size, epoch=epoch):
                            by_rank = [batches(size, batch_size, rank,
                                               world_size=world_size, epoch=epoch)
                                       for rank in range(world_size)]
                            values = [index for _, rank_batches in by_rank
                                      for batch in rank_batches for index in batch]
                            counts = Counter(values)
                            padded_size = math.ceil(size / world_size) * world_size
                            self.assertEqual(set(counts), set(range(size)))
                            self.assertEqual(len(values), padded_size)
                            self.assertEqual(sum(count - 1 for count in counts.values()),
                                             padded_size - size)
                            expected_lengths = [len(batch) for batch in by_rank[0][1]]
                            for sampler, rank_batches in by_rank:
                                self.assertEqual(len(sampler), len(rank_batches))
                                self.assertEqual([len(batch) for batch in rank_batches],
                                                 expected_lengths)
                                self.assertTrue(all(0 < len(batch) <= batch_size
                                                    for batch in rank_batches))

    def test_80k_on_four_ranks_visits_each_row_once_with_834_updates_per_epoch(self):
        size, world_size, effective_batch_size = 80000, 4, 96
        for batch_size in (1, 2, 4):
            accumulation = effective_batch_size // (world_size * batch_size)
            for epoch in (0, 1):
                with self.subTest(batch_size=batch_size, epoch=epoch):
                    values = []
                    for rank in range(world_size):
                        sampler, rank_batches = batches(
                            size, batch_size, rank, world_size=world_size, epoch=epoch,
                        )
                        self.assertEqual(len(sampler), 20000 // batch_size)
                        self.assertEqual(len(rank_batches), len(sampler))
                        self.assertTrue(all(len(batch) == batch_size for batch in rank_batches))
                        self.assertEqual(math.ceil(len(sampler) / accumulation), 834)
                        # The last accumulation window has 8 samples per rank.
                        tail = rank_batches[(834 - 1) * accumulation:]
                        self.assertEqual(sum(map(len, tail)), 8)
                        values.extend(index for batch in rank_batches for index in batch)
                    self.assertEqual(Counter(values), Counter(range(size)))

    def test_odd_sized_dataset_keeps_last_source_row(self):
        # SWIFT's original floor(N / world_size) sampler permanently lost this row.
        size = 1003
        for world_size in (2, 4):
            for epoch in (0, 1):
                with self.subTest(world_size=world_size, epoch=epoch):
                    values = []
                    lengths = []
                    for rank in range(world_size):
                        sampler, rank_batches = batches(size, 2, rank,
                                                        world_size=world_size, epoch=epoch)
                        self.assertEqual(len(sampler), len(rank_batches))
                        lengths.append([len(batch) for batch in rank_batches])
                        values.extend(index for batch in rank_batches for index in batch)
                    self.assertTrue(all(rank_lengths == lengths[0] for rank_lengths in lengths))
                    self.assertEqual(set(values), set(range(size)))
                    self.assertIn(size - 1, values)
                    self.assertEqual(len(values), size + 1)

    def test_seed_and_epoch_reconstruct_the_same_order(self):
        for world_size in (2, 4):
            for rank in range(world_size):
                _, epoch_one = batches(31, 2, rank, world_size=world_size, epoch=1)
                _, reconstructed = batches(31, 2, rank, world_size=world_size, epoch=1)
                _, epoch_two = batches(31, 2, rank, world_size=world_size, epoch=2)
                self.assertEqual(epoch_one, reconstructed)
                self.assertNotEqual(epoch_one, epoch_two)

    def test_resume_skip_preserves_epoch_through_swift_dataloader(self):
        # Exercise the exact loader.set_epoch path used after Trainer resumes.
        # The original SkipBatchSampler swallowed that call and reused epoch 0.
        for world_size in (2, 4):
            for rank in range(world_size):
                for epoch in (1, 3):
                    with self.subTest(world_size=world_size, rank=rank, epoch=epoch):
                        _, full = batches(47, 2, rank, world_size=world_size, epoch=epoch)
                        sampler = GroundingBatchSampler(
                            47, 2, num_replicas=world_size, rank=rank,
                            seed=42, shuffle=True, skip_batches=2,
                        )
                        loader = DataLoaderShard(
                            list(range(47)), batch_sampler=sampler,
                            device=torch.device("cpu"), num_workers=0,
                        )
                        loader.set_epoch(epoch)
                        resumed = [batch.tolist() for batch in loader]
                        self.assertEqual(resumed, full[2:])
                        self.assertEqual(len(loader), len(full) - 2)

    def test_skipping_a_completed_epoch_yields_no_more_samples(self):
        for world_size in (2, 4):
            for rank in range(world_size):
                _, full = batches(9, 2, rank, world_size=world_size, epoch=1)
                sampler, remaining = batches(9, 2, rank, world_size=world_size,
                                             epoch=1, skip=len(full))
                self.assertEqual(remaining, [])
                self.assertEqual(len(sampler), 0)


if __name__ == "__main__":
    unittest.main()
