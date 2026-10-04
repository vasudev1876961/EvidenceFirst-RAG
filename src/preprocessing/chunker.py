"""Section-aware semantic chunker for scientific papers."""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from src.models import Paper, SemanticChunk
from src.preprocessing.cleaner import clean_academic_text

# Common CV Benchmark Datasets
KNOWN_DATASETS = [
    "DarkFace", "ExDark", "LOL", "NightOwls", "COCO", "Pascal VOC", 
    "BDD100K", "Cityscapes", "ImageNet", "SID", "KITTI"
]

# Common Computer Vision Architectures
KNOWN_ARCHITECTURES = [
    "Vision Transformer", "ViT", "Swin Transformer", "Swin", "DETR",
    "Deformable DETR", "YOLOv8", "YOLOv7", "YOLOv5", "YOLO", "Faster R-CNN",
    "RetinaNet", "ResNet-50", "ResNet", "CNN"
]

# Common Evaluation Metrics
KNOWN_METRICS = [
    "mAP@0.5:0.95", "mAP@0.5", "mAP", "AP50", "AP75", "FPS", "FLOPs", "params"
]


class SemanticChunker:
    """Splits papers into section-anchored semantic chunks with scientific entity tags."""

    def __init__(self, target_chunk_size: int = 350, chunk_overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """Identify datasets, models, and metrics mentioned in the chunk text."""
        detected_dataset = None
        for d in KNOWN_DATASETS:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                detected_dataset = d
                break

        detected_arch = None
        for a in KNOWN_ARCHITECTURES:
            if re.search(rf"\b{re.escape(a)}\b", text, re.IGNORECASE):
                detected_arch = a
                break

        detected_metric = None
        for m in KNOWN_METRICS:
            if re.search(rf"\b{re.escape(m)}\b", text, re.IGNORECASE):
                detected_metric = m
                break

        return {
            "dataset": detected_dataset,
            "model_architecture": detected_arch,
            "evaluation_metric": detected_metric
        }

    def chunk_paper(self, paper: Paper) -> List[SemanticChunk]:
        """Convert a Paper into section-preserving semantic chunks."""
        chunks: List[SemanticChunk] = []
        chunk_idx = 1

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
                        entities = self._extract_entities(chunk_text)
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
                entities = self._extract_entities(chunk_text)
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

        return chunks
