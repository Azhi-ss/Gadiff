"""
Integrate EvoDiffMol diffusion model into the PolyGA optimization pipeline.

Usage in run_optimization.py:
    from setup_diffusion import setup_diffusion_modules
    land = setup_diffusion_modules(land, checkpoint_path, device='cuda')
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def setup_diffusion_modules(
    land: Any,
    checkpoint_path: str,
    device: str = "cuda",
    ad_threshold: float = 0.4,
    ood_penalty_weight: float = 0.5,
    egd_mutation_prob: float = 0.3,
    t_prime: int = 250,
    enable_cbsg: bool = True,
) -> Any:
    """Wire EvoDiffMol diffusion modules into the PolyGA PolyLand.

    After calling this, the GA loop will:
    - Use CFM (conformation-guided fragment mutation) instead of random mutation
    - CBSG confidence-bounded surrogate guidance (optional AD filtering)

    Returns *land* with land.egd_mutator and land.ad_checker attached.

    Graceful degradation: if any module fails to load, the original GA
    runs unchanged (random mutation fallback).
    """
    logger.info("Setting up diffusion-guided GA modules")

    # 1. Applicability Domain (CBSG) -- lightweight, no GPU needed
    ad_checker = None
    if enable_cbsg:
        try:
            from polyga.applicability_domain import ApplicabilityDomain

            ref_smiles = list(land.planet.chromosomes.values())
            ad_checker = ApplicabilityDomain(
                reference_smiles=ref_smiles,
                threshold=ad_threshold,
            )
            land.ad_checker = ad_checker
            logger.info(
                "CBSG ready: %d reference fragments, threshold=%.2f",
                ad_checker.reference_count,
                ad_threshold,
            )
        except Exception:
            logger.warning("CBSG setup failed — domain filter disabled", exc_info=True)

    # 2. EGD mutator (CFM) -- requires EvoDiffMol checkpoint
    try:
        from polyga.diffusion_mutator import EGDFragmentMutator

        model = _load_evodiffmol_model(checkpoint_path, device)
        if model is None:
            logger.warning(
                "Cannot load EvoDiffMol checkpoint: %s — "
                "CFM disabled, falling back to random mutation",
                checkpoint_path,
            )
            return land

        mutator = EGDFragmentMutator(
            diffusion_model=model,
            device=device,
            t_prime=t_prime,
            ad_checker=ad_checker,
        )
        land.egd_mutator = mutator
        land.egd_mutation_prob = egd_mutation_prob
        logger.info(
            "CFM ready: checkpoint=%s, t_prime=%d, device=%s, prob=%.2f",
            os.path.basename(checkpoint_path),
            t_prime,
            device,
            egd_mutation_prob,
        )
    except Exception:
        logger.warning("CFM setup failed — falling back to random mutation", exc_info=True)

    return land


def _load_evodiffmol_model(
    checkpoint_path: str, device: str = "cuda"
) -> Optional[Any]:
    """Load EvoDiffMol diffusion model from checkpoint.

    Returns None if loading fails (caller falls back to random mutation).
    """
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return None

    try:
        import yaml
        from easydict import EasyDict

        from configs.datasets_config import get_dataset_info
        from evodiffmol.models.epsnet import get_model
    except ImportError as e:
        logger.error("EvoDiffMol import failed: %s", e)
        return None

    try:
        import torch

        # Auto-detect model config
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "configs"
        )
        config_path = os.path.join(config_dir, "general_without_h.yml")
        if not os.path.exists(config_path):
            config_path = "configs/general_without_h.yml"

        with open(config_path) as f:
            model_config = EasyDict(yaml.safe_load(f))

        dataset_name = model_config.get("dataset", "moses")
        remove_h = model_config.get("remove_h", True)
        dataset_info = get_dataset_info(dataset_name, remove_h)

        model_config.model.num_atom = len(dataset_info["atom_decoder"]) + 1
        model = get_model(model_config.model).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        logger.info("EvoDiffMol model loaded: %s", os.path.basename(checkpoint_path))
        return model

    except Exception:
        logger.exception("Failed to load EvoDiffMol model")
        return None


def create_demo_runner(
    land: Any,
    n_generations: int = 100,
    population_size: int = 180,
    random_seed: int = 42,
) -> Any:
    """Create a DEMOPolyGA runner wired to an existing PolyLand.

    This replaces the single-population loop with the three-population
    HCEA architecture. Requires land.egd_mutator already set.
    """
    try:
        from polyga.demo_polyga import DEMOPolyGA
        from polyga.saes_selector import SAESSelector
    except ImportError:
        logger.warning("DEMO/SAES not available — use single-population GA")
        return None

    egd_mutator = getattr(land, "egd_mutator", None)
    ad_checker = getattr(land, "ad_checker", None)

    saes = SAESSelector(k_neighbors=5, tau=0.05)

    demo = DEMOPolyGA(
        egd_mutator=egd_mutator,
        saes_selector=saes,
        predict_function=land.fitness_function,
        fingerprint_function=land.fingerprint_function,
        population_size=population_size,
        n_generations=n_generations,
        random_seed=random_seed,
        ad_checker=ad_checker,
        ood_penalty_weight=0.5,
    )
    return demo
