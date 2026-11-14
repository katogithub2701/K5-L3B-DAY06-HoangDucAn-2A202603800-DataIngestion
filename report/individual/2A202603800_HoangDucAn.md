# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- **Corpus:** World Bank, *International Debt Report 2026*, DOI [10.1596/978-1-4648-2355-8](https://doi.org/10.1596/978-1-4648-2355-8), giấy phép CC BY 3.0 IGO. Bản PDF: `starter/data/data.pdf`; manifest và trích dẫn bắt buộc: [`starter/data/manifest.json`](../../starter/data/manifest.json).
- **Phạm vi trang:** số trang PDF vật lý, đếm từ 1. Chỉ xét dải 25–263 (239 trang vật lý); loại trang trống 26, 106; trang phân cách 25, 135; trang chỉ tham khảo 105, 134. Còn 233 trang nội dung và cả 233 trang được index (ngưỡng ít nhất 200 ký tự/trang). Trang ngoài dải không được tạo chunk.
- **Xử lý:** pypdf trích văn bản; chuẩn hóa Unicode NFC, line ending và khoảng trắng; giữ trang, nguồn và metadata từ outline. Trang rỗng được ghi nhận; không chạy OCR. Chunk ID ổn định theo UUID5.
- **Embedding:** Sentence Transformers local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, vector chuẩn hóa, CPU, batch 64. Chọn model đa ngữ vì benchmark tiếng Việt và tài liệu tiếng Anh. Chi phí dịch vụ: $0. Thời gian chạy: đọc corpus 38.215 s, tải model 64.014 s, chunking 0.120 s, embedding 232.445 s, truy xuất 13.932 s.
- **Vector store:** PostgreSQL 16 + pgvector 0.8.7, psycopg 3.3.8, cosine distance exact scan. Dữ liệu được upsert theo strategy/document; pipeline kiểm tra số chunk sau khi nạp lại.
- **Metadata:** `document_id`, `source`, `page`; metadata bổ sung gồm `section`, `part`, `chapter`, `subsection`, `year`, `document_type`, `profile_type`, `profile`, `country` hoặc `region` khi xác định được từ PDF outline. Filter benchmark dùng `profile_type` và `country`.

## Cấu hình và benchmark

- **A:** cửa sổ 500 ký tự, overlap 50; không filter.
- **B:** gom theo đoạn, tối đa 800 ký tự; không filter.
- **C:** cùng chunk với B; áp dụng filter metadata được khai báo trong câu hỏi. Do đó B–C chỉ khác việc áp dụng filter.
- **Benchmark:** 24 câu, 16 dev / 8 test; 5 câu có filter, trong đó 2 câu test. Evidence được kiểm tra trước khi chạy; kết quả preflight có 0 lỗi và 0 cảnh báo. Câu hỏi và gold nằm trong [`starter/benchmark/benchmark.json`](../../starter/benchmark/benchmark.json).
- **Lệnh chạy:** từ `starter/`: `python -m src.run`. Cấu hình, validation, summary và trace được lưu dưới `starter/results/`.
- **Tiêu chí báo cáo:** Hit@1, Evidence Hit@3, số chunk và filter compliance. Test đã được xem trong quá trình rà soát/điều chỉnh pipeline, nên các số test dưới đây không còn là đánh giá holdout độc lập; cần benchmark test mới nếu cần kết luận holdout.

## Kết quả retrieval

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Ghi index (s) | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | dev | 5/16 (31.25%) | 8/16 (50.00%) | 1,020 | 6.348 | $0 |
| A | test | 1/8 (12.50%) | 3/8 (37.50%) | 1,020 | 6.348 | $0 |
| B | dev | 3/16 (18.75%) | 7/16 (43.75%) | 674 | 2.543 | $0 |
| B | test | 2/8 (25.00%) | 3/8 (37.50%) | 674 | 2.543 | $0 |
| C | dev | 5/16 (31.25%) | 9/16 (56.25%) | 674 | 2.627 | $0 |
| C | test | 3/8 (37.50%) | 4/8 (50.00%) | 674 | 2.627 | $0 |

| Strategy | Coherence | Metadata đủ/đúng | Filter tuân thủ | Grounding |
|---|---|---|---|---|
| A | 1.00/2 (10 mẫu) | 15/15 metadata đúng | N/A (0 câu có filter trong A) | 0:1, 1:1, 2:3; TB 1.40/2 (5 câu) |
| B | 1.40/2 (10 mẫu) | 15/15 metadata đúng | N/A (0 câu có filter trong B) | 0:2, 1:2, 2:1; TB 0.80/2 (5 câu) |
| C | 1.40/2 (10 mẫu) | 13/13 metadata đúng | 5/5 (100%) | 0:1, 1:2, 2:2; TB 1.20/2 (5 câu) |

Thời gian trong bảng là ghi index kèm kiểm tra nạp lặp lại cho từng strategy. Chấm thủ công được thực hiện theo rubric trong `GUIDE.md`. Coherence chấm trên 10 trang được lấy mẫu và grounding trên 5 câu test (Q17–Q21) mỗi cấu hình; metadata kiểm tra mọi hit trả về trong 5 câu này (43 hit: A 15, B 15, C 13). Đã lưu điểm, lý do, người chấm và phương pháp trong [`manual_coherence_samples.json`](../../starter/results/manual_coherence_samples.json), [`manual_grounding_samples.json`](../../starter/results/manual_grounding_samples.json) và [`manual_metadata_samples.json`](../../starter/results/manual_metadata_samples.json). Các điểm này là đánh giá có giới hạn trên mẫu nhỏ, không đại diện cho toàn corpus.

Ví dụ grounding: Q17 không có cấu hình nào đưa được bằng chứng cho tổng LMIC net transfers từ outflow $118.4B sang inflow $45.4B (điểm 0). Với Q20, B không truy xuất được tổng nợ và mức tăng của Ukraine (0), trong khi C dùng filter Ukraine và lấy đúng hồ sơ trang 257 với $250,257M và +29.2% (2). A chỉ có giá trị tổng quan làm tròn $250.3B nên được 1.

## Phân tích và giới hạn

### So sánh A–B và B–C; hai ví dụ truy xuất sai

- **A–B (fixed window so với paragraph):** B giảm số chunk từ 1,020 xuống 674 (33.9%) và thời gian ghi index từ 6.348 xuống 2.543 giây. Coherence mẫu tăng từ 1.00 lên 1.40/2, phù hợp với việc giữ đoạn văn trọn ý hơn. Tuy nhiên retrieval không tốt hơn đồng đều: dev Evidence Hit@3 giảm 8/16 → 7/16, test giữ 3/8; Hit@1 test tăng 1/8 → 2/8. Vì vậy B tiết kiệm dung lượng và cải thiện tính liền mạch trên mẫu, nhưng chưa chứng minh được cải thiện bao phủ bằng chứng.
- **B–C (cùng chunk, có/không metadata filter):** C chỉ thêm lọc metadata; số chunk giữ ở 674. Evidence Hit@3 tăng 7/16 → 9/16 trên dev và 3/8 → 4/8 trên test, Hit@1 test tăng 2/8 → 3/8, filter tuân thủ 5/5. Thời gian ghi index gần như tương đương (2.543 so với 2.627 giây), vì hai cấu hình dùng cùng cách chunk; filter tác động chủ yếu ở truy xuất.
- **Ví dụ sai 1 — Q17, nguồn chuẩn trang 46 của `starter/data/data.pdf`:** câu hỏi cần tổng LMIC net transfers đổi từ outflow $118.4B sang inflow $45.4B. Top-3 của A, B và C không đưa ra các giá trị tổng này mà trả về biểu đồ hoặc số liệu liên quan đến bondholders/net transfers khác; cả ba nhận grounding 0/2. Đây là thiếu bằng chứng đúng trong top-3, dù chủ đề truy xuất có liên quan.
- **Ví dụ sai 2 — Q20, nguồn chuẩn hồ sơ Ukraine trang 257 của `starter/data/data.pdf`:** B không lấy được tổng nợ và mức tăng 29.2% (grounding 0/2). A lấy được số tổng quan làm tròn $250.3B nhưng không có số hồ sơ chính xác (1/2). C áp dụng filter Ukraine và truy xuất được $250,257M cùng +29.2% (2/2). Trường hợp này minh họa filter giúp thu hẹp đúng hồ sơ khi metadata của câu hỏi và tài liệu khớp.

### Chiến lược chọn, đánh đổi, điều kiện áp dụng

Chọn **C** làm cấu hình mặc định khi câu hỏi có bộ lọc quốc gia/vùng rõ ràng và metadata hồ sơ đã được kiểm tra: cấu hình này có kết quả retrieval tốt nhất trong lần benchmark này và tuân thủ filter 5/5. Dùng **B** khi cần paragraph chunking nhưng câu hỏi không có filter đáng tin cậy; nó giảm khoảng một phần ba số chunk và đạt coherence mẫu tương đương C. **A** chỉ phù hợp làm baseline hoặc khi cần cửa sổ kích thước cố định; kết quả coherence thấp hơn và tạo nhiều chunk hơn trong corpus này. Không áp dụng filter nếu điều kiện không có trong câu hỏi hoặc metadata còn thiếu/sai, vì có thể loại bỏ bằng chứng cần thiết. Mẫu test nhỏ nên chưa đủ để kết luận C luôn tốt hơn.

### Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí

Corpus chỉ gồm một báo cáo PDF tiếng Anh; bảng hồ sơ quốc gia/vùng bị trích xuất thành văn bản phẳng nên nhãn và chunk có thể mất quan hệ giữa tiêu đề, hàng, cột. Coherence chỉ xem 10 trang; grounding chỉ chấm 5 câu test trên mỗi cấu hình; metadata kiểm 43 hit của cùng 5 câu. Các nhãn được đánh giá theo rubric do thủ công gán nhãn nên có thể sai sót. Test chỉ có 8 câu, trong đó 2 câu có filter, và đã được quan sát trong quá trình chỉnh pipeline, vì vậy không còn là holdout độc lập. Thời gian là một lần chạy, phụ thuộc máy và gồm bước ghi index/kiểm tra nạp lại; chưa có nhiều lần chạy để báo độ biến thiên. Chi phí dịch vụ embedding là $0 do chạy model local, nhưng chưa tính thời gian máy, điện năng hoặc chi phí vận hành PostgreSQL. Thống kê token hiện là ước lượng, không phải phép đo token chính xác.

### Nếu làm lại hoặc mở rộng, cải tiến gì?

Tạo trước một bộ test mới, khóa lại trước khi tinh chỉnh và tăng số câu hỏi—đặc biệt câu hỏi theo quốc gia/vùng và câu hỏi cần tổng hợp nhiều đoạn. Bổ sung tài liệu/corpus đa dạng hơn, đồng thời giữ manifest nguồn và trang cho từng gold evidence. Cải thiện trích xuất PDF bằng parser bảng hoặc biểu diễn bảng theo hàng/cột; kiểm tra riêng trang có OCR/định dạng phức tạp. Cần ít nhất hai người chấm độc lập đánh giá các mẫu ẩn danh, tính mức đồng thuận rồi phân xử bất đồng; sử dụng LLM-judge làm tham khảo. Chạy benchmark lặp lại trên cùng môi trường, báo trung bình và độ lệch thời gian, tách thời gian ingestion, embedding, indexing và retrieval; đo token thực tế và ghi rõ phần chi phí local/cloud. Cuối cùng, đánh giá filter trên cả truy vấn có metadata đầy đủ lẫn truy vấn thiếu/không chắc metadata để xác định khi nào nên bỏ filter hoặc dùng fallback.
