"""Section-aware semantic chunker for scientific papers."""

from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Optional
from src.models import Paper, SemanticChunk
from src.preprocessing.cleaner import clean_academic_text

# Common Scientific Benchmark Datasets (Computer Vision, NLP/LLMs, Biomedical)
KNOWN_DATASETS = [
    # Computer Vision / Low-Light
    "DarkFace", "ExDark", "LOL-v2", "LOL", "NightOwls", "COCO", "Pascal VOC", 
    "BDD100K", "Cityscapes", "ImageNet", "SIDD", "SID", "KITTI", "SICE", "SMID",
    # NLP / Language Models / RAG
    "HotpotQA", "MultiHop", "SQuAD", "GSM8K", "MMLU", "MS-MARCO", "Natural Questions", 
    "TriviaQA", "NarrativeQA", "RAGBench",
    # Biomedical / Clinical
    "PubMedQA", "BioASQ", "MedQA", "MIMIC-III", "MedMCQA", "ClinicalQA",
]

# Common Scientific Architectures & Foundation Models
KNOWN_ARCHITECTURES = [
    # Computer Vision
    "Vision Transformer", "ViT", "Swin Transformer", "Swin", "DETR",
    "Deformable DETR", "RT-DETR", "YOLOv8", "YOLOv7", "YOLOv5", "YOLO", 
    "Faster R-CNN", "RetinaNet", "ResNet-50", "ResNet", "ConvNeXt", "Diffusion", "CNN",
    # NLP / Large Language Models
    "LLaMA-3", "LLaMA-2", "LLaMA", "Mistral", "GPT-4", "GPT-3.5", "Claude", 
    "BERT", "RoBERTa", "T5",
    # Biomedical
    "PubMedBERT", "BioBERT", "ClinicalBERT", "Med-PaLM",
]

# Common Scientific Evaluation Metrics
KNOWN_METRICS = [
    # Computer Vision
    "mAP@0.5:0.95", "mAP@0.5", "mAP", "AP50", "AP75", "FPS", "GFLOPs", "FLOPs", 
    "params", "PSNR", "SSIM", "LPIPS",
    # NLP & QA
    "Exact Match", "EM", "F1", "BLEU", "ROUGE-L", "ROUGE-1", "ROUGE", "Perplexity", "Accuracy",
    # Efficiency
    "Latency", "Throughput", "VRAM",
]


class SemanticChunker:
    """Splits papers into section-anchored semantic chunks with scientific entity tags."""

    def __init__(self, target_chunk_size: int = 350, chunk_overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_entities(self, text: str, paper_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        """Identify datasets, models, and metrics mentioned in the chunk text, falling back to paper metadata."""
        paper_metadata = paper_metadata or {}

        detected_dataset = None
        for d in KNOWN_DATASETS:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                detected_dataset = d
                break
        if not detected_dataset and "dataset" in paper_metadata:
            detected_dataset = paper_metadata["dataset"]

        detected_arch = None
        for a in KNOWN_ARCHITECTURES:
            if re.search(rf"\b{re.escape(a)}\b", text, re.IGNORECASE):
                detected_arch = a
                break
        if not detected_arch and "model_architecture" in paper_metadata:
            detected_arch = paper_metadata["model_architecture"]

        detected_metric = None
        for m in KNOWN_METRICS:
            if re.search(rf"\b{re.escape(m)}\b", text, re.IGNORECASE):
                detected_metric = m
                break
        if not detected_metric and "evaluation_metric" in paper_metadata:
            detected_metric = paper_metadata["evaluation_metric"]

        return {
            "dataset": detected_dataset,
            "model_architecture": detected_arch,
            "evaluation_metric": detected_metric
        }

    def chunk_paper(self, paper: Paper) -> List[SemanticChunk]:
        """Convert a Paper into section-preserving semantic chunks, including tables and figures."""
        chunks: List[SemanticChunk] = []
        chunk_idx = 1
        meta = paper.metadata or {}

        # 1. Chunk structured sections
        for section_name, raw_content in paper.sections.items():
            content = clean_academic_text(raw_content)
            if not content or len(content.split()) < 20:
                continue

            # Split section by paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            current_words: List[str] = []
            
            for para in paragraphs:
                words = para.split()
                if len(current_words) + len(words) <= self.target_chunk_size:
                    current_words.extend(words)
                else:
                    if current_words:
                        chunk_text = " ".join(current_words)
                        entities = self._extract_entities(chunk_text, meta)
                        chunk_id = f"{paper.paper_id}_c{chunk_idx:02d}"
                        chunks.append(
                            SemanticChunk(
                                chunk_id=chunk_id,
                                paper_id=paper.paper_id,
                                paper_title=paper.title,
                                section=section_name,
                                year=paper.year,
                                text=chunk_text,
                                dataset=entities["dataset"],
                                model_architecture=entities["model_architecture"],
                                evaluation_metric=entities["evaluation_metric"],
                                metadata={"word_count": len(current_words)}
                            )
                        )
                        chunk_idx += 1
                        
                        # Preserve overlap
                        overlap_words = current_words[-self.chunk_overlap:] if self.chunk_overlap < len(current_words) else []
                        current_words = overlap_words + words
                    else:
                        current_words = words

            # Remainder chunk
            if current_words and len(current_words) > 20:
                chunk_text = " ".join(current_words)
                entities = self._extract_entities(chunk_text, meta)
                chunk_id = f"{paper.paper_id}_c{chunk_idx:02d}"
                chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id,
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        section=section_name,
                        year=paper.year,
                        text=chunk_text,
                        dataset=entities["dataset"],
                        model_architecture=entities["model_architecture"],
                        evaluation_metric=entities["evaluation_metric"],
                        metadata={"word_count": len(current_words)}
                    )
                )
                chunk_idx += 1

        # 2. Chunk structured tables if present
        if getattr(paper, "tables", None):
            for t_idx, tbl in enumerate(paper.tables, 1):
                tbl_title = tbl.get("title", f"Table {t_idx}")
                tbl_content = tbl.get("content") or tbl.get("caption") or json.dumps(tbl.get("data", {}))
                table_text = f"Empirical Data Table: {tbl_title}\nContent: {tbl_content}"
                entities = self._extract_entities(table_text, meta)
                chunk_id = f"{paper.paper_id}_tbl{t_idx:02d}"
                chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id,
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        section="Tables & Benchmarks",
                        year=paper.year,
                        text=table_text,
                        dataset=entities["dataset"],
                        model_architecture=entities["model_architecture"],
                        evaluation_metric=entities["evaluation_metric"],
                        metadata={"type": "table", "title": tbl_title}
                    )
                )

        # 3. Chunk structured figures if present
        if getattr(paper, "figures", None):
            for f_idx, fig in enumerate(paper.figures, 1):
                fig_title = fig.get("title", f"Figure {f_idx}")
                fig_caption = fig.get("caption", "")
                if fig_caption:
                    fig_text = f"Experimental Figure: {fig_title}\nCaption: {fig_caption}"
                    entities = self._extract_entities(fig_text, meta)
                    chunk_id = f"{paper.paper_id}_fig{f_idx:02d}"
                    chunks.append(
                        SemanticChunk(
                            chunk_id=chunk_id,
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            section="Figures & Visualizations",
                            year=paper.year,
                            text=fig_text,
                            dataset=entities["dataset"],
                            model_architecture=entities["model_architecture"],
                            evaluation_metric=entities["evaluation_metric"],
                            metadata={"type": "figure", "title": fig_title}
                        )
                    )

        return chunks

