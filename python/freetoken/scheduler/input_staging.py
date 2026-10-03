"""Pinned descriptors and two GPU input buffers for overlapping forwards."""

import torch

from freetoken.kernel.runtime_batch import runtime_batch_module


class BatchInputStager:
    def __init__(self, device, max_requests, max_tokens):
        self.prepare_kernel = runtime_batch_module().prepare
        self.slots = []
        self.next_slot = 0
        for _ in range(2):
            host = torch.empty((max_requests, 6), dtype=torch.int32, pin_memory=True)
            self.slots.append(dict(
                host=host, host_view=host.numpy(),
                desc=torch.empty((max_requests, 6), dtype=torch.int32, device=device),
                ids=torch.empty(max_tokens, dtype=torch.int32, device=device),
                loc=torch.empty(max_tokens, dtype=torch.int32, device=device),
                positions=torch.empty(max_tokens, dtype=torch.int32, device=device),
                in_rows=torch.empty(max_tokens, dtype=torch.int64, device=device),
                in_cols=torch.empty(max_tokens, dtype=torch.int64, device=device),
                out_rows=torch.empty(max_requests, dtype=torch.int64, device=device),
                out_cols=torch.empty(max_requests, dtype=torch.int64, device=device),
                linear=torch.empty(max_requests, dtype=torch.int32, device=device),
                decode_rows=torch.empty((max_requests, 2), dtype=torch.int32, device=device),
                copied=torch.cuda.Event(), released=torch.cuda.Event(), used=False, in_use=False,
            ))

    def prepare(self, batch, page_table, token_pool, *, hybrid=False, padding_slot=0):
        slot = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % len(self.slots)
        if slot["in_use"]:
            raise RuntimeError("batch staging buffer was not released after its forward")
        if slot["used"]:
            # Host writes must not race DMA; GPU writes wait for the prior consumer.
            if not slot["copied"].query():
                slot["copied"].synchronize()
            torch.cuda.current_stream().wait_event(slot["released"])
        offset = 0
        descriptors = []
        max_extend = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            if (not 0 <= req.table_idx < page_table.shape[0]
                    or not 0 <= req.cached_len < req.device_len <= page_table.shape[1]
                    or length != req.device_len - req.cached_len):
                raise ValueError("batch descriptor exceeds its token/page table")
            linear = (req.linear_slot_idx if req.linear_slot_idx is not None else padding_slot) if hybrid else req.table_idx
            descriptors.append((req.table_idx, req.cached_len, req.device_len,
                                req.device_len if req.can_decode else -1, linear, offset))
            offset += length
            max_extend = max(max_extend, length)
        bs = len(descriptors)
        if bs > slot["desc"].shape[0] or offset > slot["ids"].numel():
            raise ValueError("batch exceeds its input staging capacity")
        slot["host_view"][:bs] = descriptors
        desc = slot["desc"][:bs]
        desc.copy_(slot["host"][:bs], non_blocking=True)
        slot["copied"].record()
        real = batch.size
        batch.input_ids = slot["ids"][:offset]
        batch.out_loc = slot["loc"][:offset]
        batch.positions = slot["positions"][:offset]
        batch.native_linear_slots = slot["linear"][:bs]
        batch.native_requests = desc
        batch.native_decode_requests = slot["decode_rows"][:bs] if batch.is_decode else None
        in_tuple = slot["in_rows"][:offset], slot["in_cols"][:offset]
        out_tuple = slot["out_rows"][:real], slot["out_cols"][:real]
        self.prepare_kernel(desc, page_table, token_pool, batch.input_ids, batch.out_loc,
                            batch.positions, *in_tuple, *out_tuple, batch.native_linear_slots,
                            slot["decode_rows"][:bs], max_extend)
        slot["in_use"] = True
        batch.native_input_slot = slot
        return in_tuple, out_tuple

    def release(self, batch):
        slot = batch.native_input_slot
        slot["released"].record()
        slot["in_use"] = False
        slot["used"] = True
