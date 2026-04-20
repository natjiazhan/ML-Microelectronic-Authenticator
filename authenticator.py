import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



DIRECTION_ORDER = ("E", "N", "S", "W")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class AuthDecision:
    predicted_id: Optional[str]
    accepted: bool
    score: float
    threshold: float
    top_k: List[Tuple[str, float]]


class SmallEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def group_query_images(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        stem = path.stem
        sample_id = stem
        for direction in DIRECTION_ORDER:
            idx = stem.find(direction)
            if idx > 0:
                prefix = stem[:idx]
                suffix = stem[idx + 1 :]
                if suffix.isdigit():
                    sample_id = prefix
                    break
        groups.setdefault(sample_id, []).append(path)

    for sample_id in groups:
        groups[sample_id] = sorted(groups[sample_id])
    return groups


def load_reference_database(reference_root: Path) -> Dict[str, List[Path]]:
    if not reference_root.exists():
        raise FileNotFoundError(f"Reference database folder not found: {reference_root}")

    database: Dict[str, List[Path]] = {}

    for item in sorted(reference_root.iterdir()):
        if item.is_file() and is_image_file(item):
            stem = item.stem
            component_id = stem
            for direction in DIRECTION_ORDER:
                idx = stem.find(direction)
                if idx > 0:
                    prefix = stem[:idx]
                    suffix = stem[idx + 1 :]
                    if suffix.isdigit() or (suffix.endswith("P") and suffix[:-1].isdigit()):
                        component_id = prefix
                        break
            database.setdefault(component_id, []).append(item)

    if not database:
        raise ValueError("No reference images found in the reference folder.")

    return database


class UPAuthenticator:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_path(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        return tensor

    @torch.no_grad()
    def embed_paths(self, image_paths: Sequence[Path]) -> torch.Tensor:
        tensors = [self.preprocess_path(path) for path in image_paths]
        batch = torch.stack(tensors, dim=0).to(self.device)
        embeddings = self.model.forward_once(batch)
        return embeddings

    @torch.no_grad()
    def build_reference_index(self, database: Dict[str, List[Path]]) -> Dict[str, torch.Tensor]:
        index: Dict[str, torch.Tensor] = {}
        for component_id, image_paths in database.items():
            embeddings = self.embed_paths(image_paths)
            prototype = embeddings.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            index[component_id] = prototype.cpu()
        return index

    @torch.no_grad()
    def authenticate_group(
        self,
        query_paths: Sequence[Path],
        reference_index: Dict[str, torch.Tensor],
        threshold: float = 0.80,
        top_k: int = 5,
    ) -> AuthDecision:
        query_embeddings = self.embed_paths(query_paths).cpu()
        query_prototype = query_embeddings.mean(dim=0)
        query_prototype = F.normalize(query_prototype, p=2, dim=0)

        scores: List[Tuple[str, float]] = []
        for component_id, reference_embedding in reference_index.items():
            score = F.cosine_similarity(
                query_prototype.unsqueeze(0),
                reference_embedding.unsqueeze(0),
            ).item()
            scores.append((component_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = scores[0]
        accepted = best_score >= threshold

        return AuthDecision(
            predicted_id=best_id if accepted else None,
            accepted=accepted,
            score=best_score,
            threshold=threshold,
            top_k=scores[:top_k],
        )


def load_model(weights_path: Optional[Path], embedding_dim: int = 128) -> nn.Module:
    model = SmallEmbeddingCNN(embedding_dim=embedding_dim)

    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for key, value in state.items():
            cleaned[key.replace("module.", "")] = value
        model.load_state_dict(cleaned, strict=True)

    return model


def save_reference_index(index: Dict[str, torch.Tensor], output_path: Path) -> None:
    serializable = {key: value.numpy() for key, value in index.items()}
    np.savez_compressed(output_path, **serializable)


def load_reference_index(index_path: Path) -> Dict[str, torch.Tensor]:
    data = np.load(index_path)
    return {key: torch.from_numpy(data[key]).float() for key in data.files}


def collect_query_paths(query: Path) -> List[Path]:
    if not query.exists():
        raise FileNotFoundError(f"Query path not found: {query}")

    if query.is_file():
        if not is_image_file(query):
            raise ValueError(f"Query file is not a supported image: {query}")
        return [query]

    paths = sorted([p for p in query.iterdir() if p.is_file() and is_image_file(p)])
    if not paths:
        raise ValueError(f"No supported images found in query folder: {query}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate UP dot images against a reference database."
    )
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Folder containing processed reference images.")
    parser.add_argument("--query", type=Path, required=True,
                        help="Either a folder of query images or a single query image.")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to trained Siamese/embedding model weights (.pt or .pth).")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Similarity threshold for match / no-match.")
    parser.add_argument("--index-out", type=Path, default=None,
                        help="Optional .npz file path to save computed reference embeddings.")
    parser.add_argument("--index-in", type=Path, default=None,
                        help="Optional .npz file path to load precomputed reference embeddings.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of best matches to report.")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to save results as JSON.")
    return parser.parse_args()


def run_authentication(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    model = load_model(args.weights)
    authenticator = UPAuthenticator(model=model)

    if args.index_in is not None:
        reference_index = load_reference_index(args.index_in)
    else:
        database = load_reference_database(args.reference_root)
        reference_index = authenticator.build_reference_index(database)
        if args.index_out is not None:
            save_reference_index(reference_index, args.index_out)

    query_paths = collect_query_paths(args.query)
    grouped_queries = group_query_images(query_paths)

    results: Dict[str, Dict[str, object]] = {}
    for sample_id, paths in grouped_queries.items():
        decision = authenticator.authenticate_group(
            query_paths=paths,
            reference_index=reference_index,
            threshold=args.threshold,
            top_k=args.top_k,
        )
        results[sample_id] = {
            "query_images": [str(p) for p in paths],
            "accepted": decision.accepted,
            "predicted_id": decision.predicted_id,
            "best_score": decision.score,
            "threshold": decision.threshold,
            "top_k": [
                {"component_id": component_id, "score": score}
                for component_id, score in decision.top_k
            ],
        }

    return results


def main() -> None:
    args = parse_args()
    results = run_authentication(args)

    for sample_id, result in results.items():
        print(f"\nSample: {sample_id}")
        print(f"Accepted: {result['accepted']}")
        print(f"Predicted ID: {result['predicted_id']}")
        print(f"Best score: {result['best_score']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print("Top matches:")
        for entry in result["top_k"]:
            print(f"  {entry['component_id']}: {entry['score']:.4f}")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {args.json_out}")


if __name__ == "__main__":
    main()
