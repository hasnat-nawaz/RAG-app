import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # kills huggingface_hub download/load bars

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Belt-and-suspenders: some transformers versions ignore the env var above
import transformers
transformers.utils.logging.disable_progress_bar()

## MAIN SCRIPT START

import time
import io
import re
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

    @staticmethod
    def clean_markdown(md_text: str) -> str:
     
        # Remove HTML comments (e.g., <!-- image -->)
        cleaned_text = re.sub(r'<!--.*?-->', '', md_text, flags=re.DOTALL)
        
        # Remove trailing whitespaces on each individual line
        cleaned_text = re.sub(r'[ \t]+$', '', cleaned_text, flags=re.MULTILINE)
        
        # Collapse 3 or more consecutive newlines into exactly 2 newlines (a single blank line)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Strip any remaining leading or trailing whitespace from the entire document
        return cleaned_text.strip()

    def load_as_markdown(self, file_path: str | Path) -> str:
        path = Path(file_path)
        result = self.converter.convert(path)
        markdown = result.document.export_to_markdown()
        print("cleaing doc...")
        cleaned_markdown = DocumentLoader.clean_markdown(markdown)
        print("cleaning done...")
        return cleaned_markdown

        

if __name__ == "__main__":
    print("Initializing DocumentLoader...")
    loader = DocumentLoader()
    input_dir = Path("../storage/uploaded_docs")
    docs = ["Hasnat_Nawaz.pdf"]
    
    # Specify your custom output directory and output file name
    output_dir = Path("../storage/output_texts/cleaned")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            start_time = time.perf_counter()
            output_file_name = Path(doc).stem + ".md"
            output_file_path = output_dir / output_file_name
            print(f"Converting '{doc}' to Markdown...")
            markdown_output = loader.load_as_markdown(input_dir / doc)
            output_file_path.write_text(markdown_output, encoding="utf-8")
            print(f"Successfully saved extracted text to: {output_file_path}")
            print(f"Time Taken : {time.perf_counter()-start_time}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
