import time
import io
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocumentStream

def create_warmup_pdf():
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Length 44 >>\n"
        b"stream\n"
        b"BT /F1 12 Tf 100 700 Td (Warmup document) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000250 00000 n \n"
        b"trailer\n"
        b"<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n"
        b"342\n"
        b"%%EOF"
    )

    return io.BytesIO(pdf)

class DocumentLoader:
    def __init__(self):
        print("1. Configuring pipeline options...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # 2. Warmup pass using a simple string stream
        print("2. Warming up layout models in memory...")
        warmup_start = time.perf_counter()
        
        dummy_stream = DocumentStream(
            name="warmup.pdf", 
            stream=create_warmup_pdf()
        )

        self.converter.convert(dummy_stream)

        print(f"3. Models loaded in {time.perf_counter() - warmup_start:.2f} seconds!")

    def load_as_markdown(self, file_path: str | Path) -> str:
        path = Path(file_path)
        result = self.converter.convert(path)
        return result.document.export_to_markdown()
        

if __name__ == "__main__":
    print("Initializing DocumentLoader...")
    loader = DocumentLoader()
    input_dir = Path("../storage/uploaded_docs")
    docs = ["Hasnat_Nawaz.pdf", "python.pdf"]
    
    # Specify your custom output directory and output file name
    output_dir = Path("../storage/output_texts/ocr")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            start_time = time.perf_counter()
            output_file_name = Path(doc).stem + ".md"
            output_file_path = output_dir / output_file_name
            print(f"Converting '{doc}' to Markdown...")
            markdown_output = loader.load_as_markdown(input_dir / doc)
            output_file_path.write_text(markdown_output, encoding="utf-8")
            print(f"\nSuccessfully saved extracted text to: {output_file_path}")
            print(f"Time Taken : {time.perf_counter()-start_time}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
