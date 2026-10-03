# Các tối ưu thử nghiệm bổ sung cho FreeToken

Tài liệu bổ sung của workspace này, cập nhật ngày **2026-10-03**. Dự án nền là
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), bản checkout
`0d652e73a452d014ac5441a15baa75348e9fcb0a`. README, hướng dẫn và tri thức của dự án
gốc tiếp tục được trình bày trong [README.md](README.md) và [docs/](docs/).

FreeToken, kiến trúc serving, các backend và mã nguồn có sẵn là công của **FreeToken
Authors và các contributor của dự án**. Các thay đổi dưới đây được phát triển và
thử nghiệm cục bộ trong quá trình người dùng làm việc với Codex. Tài liệu chỉ ghi
nhận phần bổ sung và số đo của workspace; không nhận quyền tác giả đối với nền
tảng hay thuật toán kế thừa, và không đại diện cho một release chính thức của
FlashML-org. Repository riêng cho các tối ưu này là
[tienkhoina/FreeToken-Optimization](https://github.com/tienkhoina/FreeToken-Optimization);
các thay đổi chưa được upstream tiếp nhận.

## Nguồn gốc và ghi nhận tác giả

Giấy phép của FreeToken được giữ tại [LICENSE](LICENSE), với thông báo
`Copyright 2026 FreeToken Authors`. Copyright, SPDX và dẫn nguồn có sẵn trong
các file tiếp tục xác định nguồn gốc của từng thành phần. Tài liệu bổ sung này
không thay thế giấy phép hoặc các thông báo đó.

| Thành phần kế thừa | Nguồn được ghi nhận | Phần bổ sung trong workspace |
|---|---|---|
| Loader, scheduler, expert offload, cache, API và model backends | [FreeToken](README.md) | Tối ưu operator, copy, metadata và thêm các chế độ thử nghiệm trên engine sẵn có |
| Gated DeltaNet / Flash Linear Attention | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention); header ghi Songlin Yang, Yu Zhang trong [chunk.py](python/freetoken/kernel/fla/chunk.py) | Metadata và chunk plan cho graph, mask/padding và snapshot state; giữ thuật toán recurrence được kế thừa |
| QSA sparse attention | [vLLM](https://github.com/vllm-project/vllm); SPDX và attribution trong [score.py](python/freetoken/kernel/triton/qsa/score.py) | Hỗ trợ metadata động của bucket và bảo vệ ghi KV/ring khi có padding |
| Causal convolution | sgl_kernel, với nguồn được ghi trong [causal_conv1d.py](python/freetoken/kernel/causal_conv1d.py) | Tích hợp vào đường graph/state hiện có; giữ ghi nhận kernel được mượn |
| Một số kernel GGUF | vLLM và llama.cpp, được dẫn ngay trong [mmq.cuh](python/freetoken/kernel/csrc/gguf/mmq.cuh) | Dùng để kiểm movement/operator của các định dạng; đây vẫn là engine FreeToken |
| Checkpoint và quantization | Tác giả model, nhà phát hành checkpoint, cùng PyTorch/Triton/CUDA và các thư viện backend | Tối ưu cách thực thi; không tạo hoặc nhận quyền sở hữu trọng số model |
| Giao diện chat | [Open WebUI](https://github.com/open-webui/open-webui) | Cấu hình kết nối và kiểm thử tích hợp với API của FreeToken |

Những tính năng như offload, stream/event overlap, paged KV, inline dequantization
và backend quantized đã có phần nền trong repo. Phần đóng góp của workspace là
các thay đổi cụ thể, có file và bằng chứng dưới đây. Khi chia sẻ bản sửa, cần
giữ nguồn, giấy phép và lịch sử thay đổi để người đọc phân biệt được hai phần.

## Phạm vi thay đổi

| Nhóm | Thay đổi cục bộ | Mã và tài liệu |
|---|---|---|
| CUDA/Triton MoE | Tối ưu load/decode FP4, giảm tensor input/ID trung gian, gộp reduction và activation trong những đường đã kiểm; giữ thứ tự rounding khi yêu cầu bit exact | [mxfp4_moe.py](python/freetoken/kernel/triton/mxfp4_moe.py), [moe_impl.py](python/freetoken/kernel/moe_impl.py) |
| NVFP4 | Điều chỉnh geometry của decode theo shape; hit/fetch giữ output theo route rồi dùng một FP32 reduction có thứ tự, tránh cộng hai partial sum đã làm tròn BF16 | [fused_nvfp4.py](python/freetoken/moe/fused_nvfp4.py), [route merge benchmark](benchmarks/bench_nvfp4_route_merge.py) |
| Full-bank copy | `ByteCopyPlan` chuẩn bị descriptor một lần và gửi nhiều memcpy bằng một lời gọi C++; thay vòng dispatch tensor copy ở prefill | [copy_plan.py](python/freetoken/kernel/copy_plan.py), [copy_plan.cuh](python/freetoken/kernel/csrc/jit/copy_plan.cuh) |
| Indexed copy | Tối ưu copy nguyên byte và thêm lựa chọn phân block theo kích thước bank; có cả kết quả thắng và thua khi copy chồng với compute | [fast_index_copy_scheduled.cuh](python/freetoken/kernel/csrc/jit/fast_index_copy_scheduled.cuh), [benchmark](benchmarks/bench_copy_kernel_schedule.py) |
| Hot expert | Policy `adaptive_hot` học routing online, bảo vệ một phần cache và decay score; không thay router để ép chọn expert đang resident | [adaptive_hot.py](python/freetoken/moe/adaptive_hot.py), [adaptive_hot_cache.cuh](python/freetoken/kernel/csrc/jit/adaptive_hot_cache.cuh) |
| Khởi tạo expert cache | `--moe-cache-init random` nạp expert sau warmup/capture và trước request đầu; seed preload độc lập với sampling RNG | [offload_cache.py](python/freetoken/moe/offload_cache.py), [engine.py](python/freetoken/engine/engine.py) |
| Native CPU/GPU scheduling | C++/CUDA phân hit/miss, lập descriptor copy và chọn nhánh CPU/GPU; graph chứa các phụ thuộc stream/event, profile chọn số miss gửi GPU | [native_schedule.py](python/freetoken/moe/native_schedule.py), [hướng dẫn calibration](benchmarks/README_native_moe_schedule.md) |
| Input, KV metadata và sampling | Descriptor pinned, buffer tái sử dụng, chuẩn bị vị trí/page trên GPU, giảm Python/ATen dispatch và allocation; greedy argmax int32 | [runtime batch](benchmarks/README_runtime_batch.md), [decode metadata](benchmarks/README_decode_metadata.md) |
| Qwen prefill | Graph cho QSA/GDN/PLE; chế độ exact, bucket, block và layer; độ dài thật/mask bảo vệ KV, recurrence, history và snapshot | [qwen_prefill.py](python/freetoken/engine/qwen_prefill.py), [bucket/block](benchmarks/README_bucket_block_prefill.md) |
| Layer prefill | C++ gọi graph theo layer/block, giữ bank hiện tại qua nhiều block, prefetch bank kế tiếp; activation dùng GPU hoặc pinned RAM theo ngân sách | [qwen_layer_prefill.py](python/freetoken/engine/qwen_layer_prefill.py), [layer_prefill.cuh](python/freetoken/kernel/csrc/jit/layer_prefill.cuh) |
| Serving cục bộ | Service FreeToken, bốn bucket cố định và Open WebUI qua API OpenAI; giữ model/graph giữa các request, bỏ audit benchmark theo token | [triển khai](../deploy/qwen-webui/README.md) |

Đường copy đã được kiểm với BF16, FP16, FP8 block, MXFP4, NVFP4, DS-FP4 và Q4_0.
Điều này không đồng nghĩa mọi kernel toán của từng định dạng đều đã được tối ưu
hoặc kiểm trên full model. Layout NVFP4 donor/Marlin/b12x có phép kiểm copy riêng;
copy đúng byte không thay thế kiểm độ đúng của backend compute.

Các lựa chọn mới căn cứ vào shape, dtype, layout và ngân sách bộ nhớ. Tên CPU/GPU
trong báo cáo chỉ mô tả máy đo; các số slot, bucket và tỷ lệ CPU/GPU cần đo lại
trên máy khác. ROCm và các backend chưa chạy không được coi là đã kiểm chứng.

## Cách kiểm chứng

Operator benchmark lưu tensor đầu vào, output tham chiếu, output sau sửa, sample
thời gian và hash source. Copy/metadata/operator yêu cầu bit exact được so dưới
dạng byte, gồm các vùng canary và state khi phép kiểm áp dụng. Compile/warmup và
khôi phục trạng thái nằm ngoài vùng đo kernel. Các số host enqueue, CUDA event,
allocator peak và throughput API được báo riêng theo ý nghĩa của chúng.

Full-model benchmark lưu lệnh, checkpoint, prompt, token IDs, logits, cache/KV,
graph và điều kiện lặp. Baseline của nhiều vòng là **snapshot workspace đã có
tối ưu trước đó**, không phải upstream nguyên bản. Không nhân các tỷ lệ tăng tốc
ở những vòng khác nhau để suy ra một tỷ lệ tổng thể.

Thay bucket, GEMM shape hoặc cách chia recurrence có thể thay rounding. Graph
replay so với eager cùng physical shape có kiểm byte/state riêng; kết quả giữa
những shape khác nhau được báo bằng sai số. Token sinh giống nhau chưa đủ chứng
minh logits hoặc toàn bộ forward bit exact. Nhánh CPU dùng mức sai khác số thực
được chấp nhận, và chỉ nên bật khi calibration cho thấy có lợi.

Máy đo chính: RTX A5000 24 GiB, RAM khoảng 125,9 GiB, 64 logical CPU trong affinity,
PyTorch 2.11/CUDA 13 và Triton 3.6. Chưa có số đo đủ để suy rộng sang mọi thiết bị.

## Kết quả đã lưu

### Kernel, copy và overhead

| Phép đo | Trước → sau | Điều kiện và ý nghĩa |
|---|---|---|
| MXFP4 experts decode | 0,148096 → 0,134400 ms | M=1, H=I=2.880, top-k=4; graph đã warm, cải thiện operator 9,25% |
| MXFP4 experts prefill | 1,661440 → 1,570800 ms | M=128, cùng H/I/top-k; cải thiện operator 5,46% |
| MoE forced miss, 4 expert | 4,430848 → 4,416512 ms | Copy RAM→VRAM chiếm phần lớn; lợi ích pipeline chỉ khoảng 0,32% |
| MXFP4 full-bank copy enqueue | 73,532 → 36,605 µs | `ByteCopyPlan` giảm dispatch host; thời gian truyền dữ liệu lớn gần như giữ nguyên |
| Synthetic runtime decode | 1,1746 → 0,5067 ms | Batch 1, context capacity 8.192; input/metadata/attention graph/sampling, không có MLP/MoE |

Nguồn: [kernel MXFP4](../result/cuda_moe_bitexact_2026-09-30/REPORT.md),
[copy plan](../result/copy_overlap_2026-09-30/REPORT.md),
[copy cùng compute](../result/copy_overlap_2026-09-30/SCHEDULE_REPORT.md),
[runtime bundle](../result/decode_runtime_bundle_2026-10-01/REPORT.md).

Vòng kernel MXFP4 lưu 389 so sánh ở 39 case, không có byte khác trong các case
đã chạy. Weighted indexed copy có tình huống nhanh hơn NVFP4/Q4_0 nhưng chậm hơn
BF16/FP16; giữ dạng lựa chọn thay vì thay default toàn repo. Runtime bundle giảm
overhead synthetic rõ, nhưng GPT-OSS full-model chỉ thay throughput khoảng
0,34–0,37%, chưa chứng minh lợi ích lớn ở tốc độ token.

### Qwen: prefill ngắn và input dài

Đợt Qwen ngắn dùng 3.072 slot, KV 8.192, cap 64, greedy 16 output token và trung
vị năm lượt lặp/chủ đề. Prefill graph kết hợp selective native loading:

| Prompt | TTFT trước → sau | Decode trước → sau |
|---|---:|---:|
| Toán | 6,007 → 1,414 s | 26,27 → 22,32 tok/s |
| Code | 6,064 → 1,214 s | 25,07 → 29,20 tok/s |

12 cặp logits prefill và chuỗi output token trùng trong đợt này. Decode tăng
hoặc giảm theo workload, nên không gọi đây là cải thiện decode đồng đều.
[Báo cáo và bằng chứng](../result/qwen38_prefill_2026-10-01/REPORT.md).

Một đợt input dài riêng dùng cache 1.024 slot, KV 131.072, block cap 128 và
8 output token: bucket đi qua toàn model nhiều lần có TTFT 270,26 s ở 10k,
layer sweep có TTFT 28,32 s. Lượt layer 10k đó được đo trước các sửa cuối; bản
cuối đã chạy 100k token mới, prefix hit 0, TTFT 310,85 s, với 3,818 GiB activation
plane trên pinned RAM. Chưa có full-model A/B 100k lạnh để tính tăng tốc.
[Điều kiện và giới hạn](../result/bucket_block_prefill_2026-10-02/SPEED_REPORT_VI.md).

Đợt 10k sau dùng cache 2.048/KV 16.384/32 output lại cho thấy eager full-layer
chunk 8.192 nhanh nhất trong các cấu hình hoàn tất: warm TTFT **14,39 s**;
full-layer chunk 128 là 457,64 s, selective eager 128 là 209,85 s. Bucket/layer
chưa hoàn tất trong ma trận mới. Vì vậy mốc layer nhanh 9,54 lần ở đợt cũ chỉ
đối chiếu với bucket cap 128 của đợt đó, không chứng minh thắng default chunk
8.192. [Ma trận đã công bố](../result/qwen_10k_matrix_2026-10-02/REPORT.md).

### Bốn bucket 128, 512, 2.048, 8.192

Cache 2.048 slot, KV/context 16.384, batch 1, mỗi prompt chạy một lượt sinh
32 token. Reset KV/prefix và cache expert về random trước từng lượt, ngoài TTFT:

| Input token | Physical bucket | TTFT | Prefill | Decode |
|---:|---:|---:|---:|---:|
| 32 | 128 | 1,68 s | 19,70 tok/s | 14,14 tok/s |
| 128 | 128 | 3,67 s | 42,02 tok/s | 16,45 tok/s |
| 1.024 | 2.048 | 6,59 s | 158,22 tok/s | 10,28 tok/s |
| 4.096 | 8.192 | 11,22 s | 374,67 tok/s | 10,17 tok/s |

Đủ bốn prefill graph và một decode graph; không capture thêm trong các request
đã đo. Đây là số đo từng request, không phải trung vị và không phải A/B eager
cùng shape. [Ba prompt](../result/qwen_bucket_128_512_2048_8192_2026-10-02/REPORT.md),
[prompt 32](../result/qwen_bucket_128_512_2048_8192_2026-10-02/REPORT_32.md).

### Expert cache ảnh hưởng decode

Chạy lại đúng source và script baseline cũ, giữ cache giữa request; KV 8.192,
context 2.048, prefill eager cap 64, một decode graph, native scheduler tắt.
Chỉ đổi cache 2.048 → 4.096 slot:

| Prompt | Decode 2.048 slot | Decode 4.096 slot | Cách đo |
|---|---:|---:|---|
| Toán | 15,34 tok/s | 27,25 tok/s | Trung vị năm lượt, 16 output token |
| Code | 21,18 tok/s | 26,90 tok/s | Trung vị năm lượt, 16 output token |
| Cùng prompt raw 32 in / 32 out | 14,84 tok/s | 20,53 tok/s | Một lượt sau cùng chuỗi toán/code |

Token IDs và logits prefill trùng ở 13 request đối chiếu. Các server được nạp
riêng; độ ấm kernel/page cache và thời điểm đo vẫn có thể khác. Tăng slot là
thay đổi ngân sách VRAM, không phải bằng chứng tăng tốc kernel. Cache expert
nóng cũng khác prefix/KV cache: những prompt ngắn này đều có prefix hit 0.
[Báo cáo 4.096 slot](../result/qwen_old_script_cache4096_2026-10-03/REPORT.md),
[cùng prompt 32](../result/qwen_old_script_cache4096_2026-10-03/REPORT_32.md).

## Các giới hạn cần giữ khi đọc kết quả

- Random preload không biết trước expert nào sẽ nóng. Routing trong request
  tiếp tục cập nhật cache; reset/rebuild mới xóa lịch sử đó.
- `adaptive_hot` có kết quả tốt ở một số nhóm và xấu lúc chuyển chủ đề; LRU
  vẫn được giữ làm lựa chọn chuẩn. [Báo cáo hot expert](../result/adaptive_hot_2026-09-30/REPORT.md).
- Copy layer kế tiếp có thể chuẩn bị sớm vì bank đã biết. Expert cụ thể của
  layer tiếp theo phải chờ router; các thử nghiệm không dùng future expert IDs.
- GPU hit có thể tính trong khi miss đang copy hoặc CPU đang chạy, nhưng
  throughput còn bị giới hạn bởi PCIe, RAM và phụ thuộc dữ liệu. Calibration
  Qwen trên máy này chọn GPU cho mọi decode miss; pool 64 CPU sẵn không có nghĩa
  CPU đang tính expert trong deployment.
- Hit-D2D giúp một đợt GPT-OSS, nhưng Qwen 10k cap 128 giảm byte copy mà TTFT
  vẫn tăng. [GPT-OSS hit-D2D](../result/prefill_hit_d2d_2026-10-01/REPORT.md).
- Bản layer prefill chặn decode admission tới khi prompt đó hoàn tất. State và
  KV vẫn có phần quản lý host; các tối ưu metadata không biến toàn scheduler
  thành native hay tự thêm RAM KV spill.
- Các test broad có lỗi equality giữa những shape khác nhau đã được tái hiện
  trên baseline; kết quả test và các case bị skip nằm trong báo cáo từng vòng.
- Checkpoint khác, multimodal, backend khác và nhiều request phải kiểm riêng.
  Dung lượng context khai báo là sức chứa, không phải số token luôn được attention.

Calibration một expert Qwen NVFP4 thật sau compile ghi CPU task 5,119 ms,
copy H2D 0,281 ms và GPU math 0,0798 ms trên máy đo. Thời gian CPU branch đầy đủ
và overhead dispatch cũng được lưu, nên không chọn tỷ lệ CPU/GPU chỉ theo tốc độ
phép nhân. [Số đo calibration](../result/qwen38_prefill_2026-10-01/REPORT.md).

## Cấu hình triển khai đã kiểm thử

[Deployment cục bộ](../deploy/qwen-webui/README.md) dùng source hiện tại, cache
2.048 slot, LRU, preload random seed 42, native GPU miss fraction 1, QSA sparse,
PLE disk, NVFP4 Triton, KV/context 16.384 và bốn bucket trên. FreeToken chạy qua
systemd user service, Open WebUI chạy Docker; integration test trong trình duyệt
đã đăng nhập, chọn model và nhận câu trả lời. Đây là kiểm thử tích hợp serving.

Bốn shape cố định được chọn bởi
[deployment hook](../deploy/qwen-webui/instrument/sitecustomize.py). CLI `bucket`
trong engine thông thường tạo các shape lũy thừa hai từ 8 tới cap; chỉ truyền
`--moe-prefill-graph-max-tokens 8192` không tự tạo đúng bốn shape x4. Runtime
reuse graph theo bucket và mask độ dài thật; padding vẫn tốn compute.

Deployment hiện đặt `max-running-requests=1`, decode graph batch 1. Nhiều HTTP
request được nhận và xếp hàng; cấu hình này chưa xử lý nhiều request đồng thời.
Tăng concurrency cần đổi hook, capture batch lớn và đo lại VRAM/throughput.

```bash
# Từ thư mục FreeToken; cần deployment cùng workspace đã cài.
systemctl --user status freetoken-qwen.service
cd ../deploy/qwen-webui
docker compose ps
curl http://127.0.0.1:8000/health
```

Thông tin đăng nhập nằm trong file riêng của deployment. Mật khẩu, token,
session trình duyệt và dữ liệu chat không đưa vào README hoặc bản chia sẻ.

## Đo lại trên máy khác

Chọn cache sau khi chừa KV, state và peak graph/workspace. Giữ cùng checkpoint,
input IDs, số output, cache state, sampling và graph khi so hai phiên bản.
Ghi riêng cold request, warm repeat và startup; không so trung vị warm với
một request đã reset cache mà bỏ qua khác biệt đó.

```bash
# Chạy từ repo; CUDA toolkit phải khớp CUDA major của PyTorch.
python benchmarks/calibrate_moe_schedule.py \
  --workload qwen3.8-flash-next --format nvfp4 --threads 0 \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --output /path/to/results/moe_profile.json
```

Chọn workload được script hỗ trợ và tham số H/I/expert tương ứng checkpoint;
xem `--help` và [hướng dẫn calibration](benchmarks/README_native_moe_schedule.md).
CPU task đã compile và GPU graph đã warm được đo riêng; mẫu tổng hợp không
thay thế route/copy của model thật. Profile của máy khác cần đo lại.

Các [benchmark kernel](benchmarks/bench_moe_bitexact.py),
[copy](benchmarks/bench_copy_overlap.py),
[runtime](benchmarks/bench_runtime_batch.py) và
[Qwen graph](benchmarks/bench_qwen_prefill_ops.py) hỗ trợ kiểm từng phần.
Chỉ công bố phạm vi đã chạy; giữ tensor, source hash, log và số liệu gốc.

## Tài liệu và bằng chứng trong workspace

[result/](../result/) lưu báo cáo, snapshot trước/sau, test XML, raw tensors,
logits, token IDs, lệnh và trace của từng vòng. Các đường dẫn `../result/` và
`../deploy/` thuộc workspace thử nghiệm, không mặc nhiên có trong repo upstream.
Khi chia sẻ tài liệu cần kèm artifacts thích hợp hoặc thay bằng liên kết công
khai; giữ nguyên attribution và tài liệu của tác giả nền.

Tài liệu này tổng hợp bằng chứng đã lưu, không chạy benchmark mới và không
thay nội dung [README gốc](README.md), [LICENSE](LICENSE),
[CONTRIBUTING.md](CONTRIBUTING.md) hay các hướng dẫn đang có.
