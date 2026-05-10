"""
MMPolymer Prediction Module for PolyGA

This module provides a prediction function interface compatible with PolyGA's
genetic algorithm framework. It wraps the MMPolymer inference pipeline for
efficient batch prediction of polymer properties.

Key Features:
- PolyGA-compatible interface (df, fp_headers, models)
- Efficient caching mechanism
- Batch processing
- Support for multiple properties (Tg, DC, Eb, etc.)

Usage in PolyGA:
    from polyga.mmpolymer_predict import create_mmpolymer_predictor
    
    predict_fn = create_mmpolymer_predictor(
        weight_dir='/internfs/Zy/dataset/finetune_data',
        properties=['Tg', 'DC']
    )
    
    planet = PolyPlanet(
        name='my_planet',
        predict_function=predict_fn,
        fingerprint_function=my_fp_fn,
        ...
    )
"""

import os
import sys
import csv
import shutil
import pickle
import lmdb
import subprocess
import torch
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
import logging
import tempfile
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings(action='ignore')
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration matching training pipeline."""
    NUM_CONFORMERS: int = 14
    ETKDG_VERSION: int = 3
    PRUNE_RMS_THRESH: float = 0.5      # RMSD threshold for pruning
    MMFF_MAX_ITERS: int = 500          # Force field optimization iterations
    CONFORMER_MAX_ATTEMPTS: int = 5000


# ============================================================================
# Molecular Processing Functions
# ============================================================================

def smi2scaffold(smi: str) -> str:
    """Generate Murcko scaffold from SMILES string."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
    except Exception as e:
        logger.debug(f"Failed to generate scaffold for {smi}: {e}")
        return smi


def smi2_2Dcoords(smi: str) -> np.ndarray:
    """Generate 2D coordinates from SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape mismatch for {smi}"
    return coordinates


def smi2_3Dcoords(smi: str, cnt: int = Config.NUM_CONFORMERS) -> List[np.ndarray]:
    """
    Generate multiple 3D conformers from SMILES using ETKDG method.
    
    Matches training pipeline for consistency.
    """
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    
    # Configure ETKDG parameters
    if Config.ETKDG_VERSION == 3:
        params = AllChem.ETKDGv3()
    elif Config.ETKDG_VERSION == 2:
        params = AllChem.ETKDGv2()
    else:
        params = AllChem.ETKDG()
    
    params.randomSeed = -1
    params.pruneRmsThresh = Config.PRUNE_RMS_THRESH
    params.useRandomCoords = True
    params.numThreads = 0
    
    for seed in range(cnt):
        try:
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol, params)
            
            if res == 0:
                # Successfully embedded, optimize with MMFF
                try:
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                        if ff is not None:
                            ff.Minimize(maxIts=Config.MMFF_MAX_ITERS)
                            coordinates = mol.GetConformer().GetPositions()
                        else:
                            AllChem.UFFOptimizeMolecule(mol, maxIts=Config.MMFF_MAX_ITERS)
                            coordinates = mol.GetConformer().GetPositions()
                    else:
                        AllChem.UFFOptimizeMolecule(mol, maxIts=Config.MMFF_MAX_ITERS)
                        coordinates = mol.GetConformer().GetPositions()
                except Exception as e:
                    logger.debug(f"Optimization failed for seed={seed}: {e}")
                    coordinates = mol.GetConformer().GetPositions()
                    
            elif res == -1:
                # ETKDG failed, try with more attempts
                logger.debug(f"ETKDG failed for seed={seed}, retrying")
                mol_tmp = Chem.MolFromSmiles(smi)
                mol_tmp = AllChem.AddHs(mol_tmp)
                
                if Config.ETKDG_VERSION == 3:
                    params_retry = AllChem.ETKDGv3()
                elif Config.ETKDG_VERSION == 2:
                    params_retry = AllChem.ETKDGv2()
                else:
                    params_retry = AllChem.ETKDG()
                    
                params_retry.randomSeed = seed
                params_retry.maxAttempts = Config.CONFORMER_MAX_ATTEMPTS
                params_retry.useRandomCoords = True
                
                res_retry = AllChem.EmbedMolecule(mol_tmp, params_retry)
                
                if res_retry == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_tmp, maxIts=Config.MMFF_MAX_ITERS)
                        coordinates = mol_tmp.GetConformer().GetPositions()
                    except:
                        coordinates = mol_tmp.GetConformer().GetPositions()
                else:
                    logger.debug(f"All 3D methods failed for seed={seed}, using 2D")
                    coordinates = smi2_2Dcoords(smi)
                    
        except Exception as e:
            logger.debug(f"3D generation failed for seed={seed}: {e}, using 2D")
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), \
            f"3D coordinates shape mismatch for {smi}"
        coordinate_list.append(coordinates.astype(np.float32))
    
    return coordinate_list


def process_polymer_smiles(psmiles: str) -> dict:
    """
    Process polymer SMILES to extract structure and generate conformers.
    
    Expects polymer SMILES with exactly 2 star atoms (*) marking connection points.
    """
    # Prevent RDKit from spawning threads inside a subprocess, which causes deadlocks
    try:
        AllChem.SetNumThreads(1)
    except:
        pass

    try:
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {psmiles}")
            return None
            
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Find star atoms (connection points)
        star_ids = [i for i, symbol in enumerate(atoms) if symbol == '*']
        if len(star_ids) != 2:
            logger.warning(f"Expected 2 star atoms, found {len(star_ids)}: {psmiles}")
            return None
        
        # Get star atom neighbors
        star_pair_list = []
        for star_id in star_ids:
            star_pair_list.append(star_id)
            neighbors = mol.GetAtomWithIdx(star_id).GetNeighbors()
            if len(neighbors) != 1:
                logger.warning(f"Star atom {star_id} has {len(neighbors)} neighbors: {psmiles}")
                return None
            star_pair_list.append(neighbors[0].GetIdx())
        
        # Replace star atoms with their neighbor atoms
        mol.GetAtomWithIdx(star_pair_list[0]).SetAtomicNum(
            mol.GetAtomWithIdx(star_pair_list[3]).GetAtomicNum())
        mol.GetAtomWithIdx(star_pair_list[2]).SetAtomicNum(
            mol.GetAtomWithIdx(star_pair_list[1]).GetAtomicNum())
        
        smi = Chem.MolToSmiles(mol)
        scaffold = smi2scaffold(smi)
        
        # Generate conformers (14 3D + 1 2D = 15 total)
        if len(mol.GetAtoms()) > 400:
            logger.warning(f"Large molecule ({len(mol.GetAtoms())} atoms), using 2D")
            coordinate_list = [smi2_2Dcoords(smi)] * 15
        else:
            coordinate_list = smi2_3Dcoords(smi, cnt=14)
            coordinate_list.append(smi2_2Dcoords(smi))
        
        # Prepare final molecule with H atoms
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Restore star atoms
        atoms[star_pair_list[0]] = '*'
        atoms[star_pair_list[2]] = '*'
        
        return {
            'atoms': atoms,
            'coordinates': coordinate_list,
            'mol': mol,
            'smi': psmiles,
            'origin_smi': psmiles,
            'star_pair': star_pair_list,
            'scaffold': scaffold,
            'target': -999  # Placeholder for inference
        }
    except Exception as e:
        logger.error(f"Failed to process PSMILES '{psmiles}': {e}")
        return None


def load_tokenizer():
    """Load PolymerSmilesTokenizer from MMPolymer."""
    # Add MMPolymer to path
    mmpolymer_path = '/root/code/MMPolymer'
    if mmpolymer_path not in sys.path:
        sys.path.insert(0, mmpolymer_path)
    
    from MMPolymer.models.PolymerSmilesTokenization import PolymerSmilesTokenizer
    return PolymerSmilesTokenizer.from_pretrained()


# ============================================================================
# Prediction Pipeline
# ============================================================================

def prepare_lmdb_data(psmiles_list: List[str], output_dir: str, tokenizer) -> int:
    """Prepare LMDB database from PSMILES list."""
    dc_data_dir = os.path.join(output_dir, 'DC')
    os.makedirs(dc_data_dir, exist_ok=True)
    
    env = lmdb.open(
        os.path.join(dc_data_dir, 'test.lmdb'),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(1000e9),
    )
    
    # Run conformer generation in parallel using all available CPUs
    # This is the CPU bottleneck, so we want max parallelism here
    n_jobs = os.cpu_count() or 1
    # Cap n_jobs to avoid overloading
    if n_jobs > 32:
        n_jobs = 32
    
    # Cap n_jobs if list is small to avoid overhead
    if len(psmiles_list) < n_jobs:
        n_jobs = len(psmiles_list)
        
    logger.info(f"Generating conformers using {n_jobs} CPU cores...")
    
    results = Parallel(n_jobs=n_jobs, backend='loky', pre_dispatch='2*n_jobs')(
        delayed(process_polymer_smiles)(psmiles)
        for psmiles in tqdm(psmiles_list, desc="Generating Conformers", disable=len(psmiles_list) < 10)
    )
    
    txn = env.begin(write=True)
    index = 0
    
    for data_info in results:
        if data_info is None:
            continue

        encoding = tokenizer(
            data_info['origin_smi'],
            add_special_tokens=True,
            max_length=411,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        data_info["input_ids"] = encoding["input_ids"].flatten()
        data_info["attention_mask"] = encoding["attention_mask"].flatten()

        txn.put(f'{index}'.encode("ascii"), pickle.dumps(data_info, protocol=-1))
        index += 1

    txn.commit()
    env.close()
    return index


def run_inference(property_name: str, cache_path: str, weight_path: str, 
                  batch_size: int = 128, use_gpu: bool = True, gpu_id: int = 0) -> Tuple[List[str], np.ndarray]:
    """Run model inference for a specific property."""
    task_name = 'DC'  # Task name is always DC for this model architecture
    head_name = property_name  # Use the actual property name for classification head
    results_path_dir = os.path.join(cache_path, f'{property_name}_results')
    os.makedirs(results_path_dir, exist_ok=True)

    # Get dict path relative to MMPolymer
    mmpolymer_dir = '/root/code/MMPolymer'
    dict_path = os.path.join(mmpolymer_dir, 'dict.txt')
    
    cmd = (
        f"python {mmpolymer_dir}/MMPolymer/infer.py "
        f"--user-dir {mmpolymer_dir}/MMPolymer "
        f"{cache_path} --task-name {task_name} "
        f"--valid-subset test --results-path {results_path_dir} "
        f"--num-workers 1 --ddp-backend=c10d --batch-size {batch_size} "
        f"--task MMPolymer_finetune --loss MMPolymer_finetune "
        f"--arch MMPolymer_base --classification-head-name {head_name} "
        f"--num-classes 1 --dict-name {dict_path} --conf-size 15 "
        f"--only-polar 1 --path {weight_path} "
        f"--log-interval 50 --log-format simple"
    )
    
    # Add GPU/CPU configuration
    if use_gpu:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + cmd
        cmd += " --fp16 --fp16-init-scale 4 --fp16-scale-window 256"
    else:
        cmd += " --cpu"
    
    # Run inference (suppress output for cleaner logs)
    result = subprocess.run(cmd, shell=True, check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Load predictions
    predict_result_path = os.path.join(results_path_dir, f"{property_name}_test_cpu.out.pkl")
    predict_outputs = pd.read_pickle(predict_result_path)
    
    # Aggregate predictions
    pred_list = []
    psmi_list = []
    for epoch_output in predict_outputs:
        pred_list.append(epoch_output['predict'])
        psmi_list.extend(epoch_output['smi_name'])
    
    pred_list = torch.cat(pred_list, dim=0).float()
    
    # Average over conformers (15 conformers per molecule)
    conf_size = 15
    total_predictions = pred_list.shape[0]
    
    if total_predictions % conf_size != 0:
        truncate_to = (total_predictions // conf_size) * conf_size
        if truncate_to == 0:
            raise ValueError("No valid predictions to aggregate")
        pred_list = pred_list[:truncate_to]
        psmi_list = psmi_list[:truncate_to]
    
    psmi_list = psmi_list[::conf_size]
    pred_list = pred_list.view(-1, conf_size).numpy().mean(axis=1)
    pred_list = np.round(pred_list, 2)
    
    return psmi_list, pred_list


# ============================================================================
# PolyGA-Compatible Predictor
# ============================================================================

class MMPolymerPredictor:
    """
    PolyGA-compatible predictor for MMPolymer.
    
    This class provides the prediction interface required by PolyGA's
    PolyPlanet framework.
    """
    
    def __init__(
        self,
        weight_dir: str,
        properties: List[str],
        batch_size: int = 128,
        use_gpu: bool = True,
        gpu_id: int = 0,
        cache_dir: Optional[str] = None,
        cleanup_cache: bool = True
    ):
        """
        Initialize MMPolymer predictor.
        
        Args:
            weight_dir: Directory containing weight files
                        Expected structure: {weight_dir}/{property}/ckpt/{property}/checkpoint_best.pt
            properties: List of properties to predict (e.g., ['Tg', 'DC'])
            batch_size: Batch size for inference
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID
            cache_dir: Directory for temporary files (None = create temp dir)
            cleanup_cache: Whether to cleanup cache after each prediction
        """
        self.weight_dir = Path(weight_dir)
        self.properties = properties
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.cleanup_cache = cleanup_cache
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix='mmpolymer_cache_'))
            self.owns_cache_dir = True
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.owns_cache_dir = False
        
        # Load tokenizer once
        logger.info("Loading PolymerSmilesTokenizer...")
        self.tokenizer = load_tokenizer()
        
        # Verify weight files
        self._verify_weights()
        
        # Prediction cache to avoid re-predicting same SMILES
        self.prediction_cache = {}
        
        logger.info(f"MMPolymerPredictor initialized (GPU: {self.use_gpu})")
    
    def _verify_weights(self):
        """Verify that all required weight files exist."""
        missing = []
        for prop in self.properties:
            weight_path = self.weight_dir / prop / "ckpt" / prop / "checkpoint_best.pt"
            if not weight_path.exists():
                missing.append(str(weight_path))
        
        if missing:
            raise FileNotFoundError(
                f"Weight files not found:\n" + "\n".join(missing)
            )
        
        logger.info(f"Verified weight files for properties: {', '.join(self.properties)}")
    
    def predict(
        self,
        df: pd.DataFrame,
        fp_headers: List[str],
        models: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Predict properties for polymers in dataframe.
        
        This is the main function called by PolyGA's prediction pipeline.
        
        Args:
            df: DataFrame with 'smiles_string' column containing polymer SMILES
            fp_headers: Fingerprint headers (not used by MMPolymer, kept for compatibility)
            models: Model dictionary (not used, kept for compatibility)
        
        Returns:
            DataFrame with predicted properties added as new columns
        """
        if 'smiles_string' not in df.columns:
            raise ValueError("DataFrame must have 'smiles_string' column")
        
        smiles_list = df['smiles_string'].tolist()
        logger.info(f"Predicting properties for {len(smiles_list)} polymers")
        
        # Check cache first
        uncached_smiles = []
        uncached_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            if smiles not in self.prediction_cache:
                uncached_smiles.append(smiles)
                uncached_indices.append(idx)
        
        if len(uncached_smiles) > 0:
            logger.info(f"Cache hits: {len(smiles_list) - len(uncached_smiles)}/{len(smiles_list)}")
            logger.info(f"Running inference for {len(uncached_smiles)} new polymers")
            
            # Run batch prediction
            new_predictions = self._batch_predict(uncached_smiles)
            
            # Update cache
            for smiles, pred in zip(uncached_smiles, new_predictions):
                self.prediction_cache[smiles] = pred
        else:
            logger.info(f"All {len(smiles_list)} predictions found in cache")
        
        # Collect all predictions - handle invalid SMILES gracefully
        for prop in self.properties:
            values = []
            for smiles in smiles_list:
                if smiles in self.prediction_cache:
                    values.append(self.prediction_cache[smiles][prop])
                else:
                    # SMILES was invalid and not cached
                    values.append(np.nan)
            df[prop] = values
        
        return df
    
    def _batch_predict(self, smiles_list: List[str]) -> List[Dict[str, float]]:
        """
        Run batch prediction on list of SMILES.
        
        Args:
            smiles_list: List of polymer SMILES strings
        
        Returns:
            List of prediction dictionaries {property: value}
        """
        # Create temporary working directory
        temp_dir = self.cache_dir / f"batch_{os.getpid()}_{len(smiles_list)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare LMDB (creates DC/test.lmdb)
            num_samples = prepare_lmdb_data(smiles_list, str(temp_dir), self.tokenizer)
            
            if num_samples == 0:
                raise ValueError("No valid SMILES in batch")
            
            if num_samples != len(smiles_list):
                logger.warning(f"Only {num_samples}/{len(smiles_list)} SMILES are valid")
            
            # Create symbolic links for each property (infer.py expects {property}/test.lmdb)
            dc_lmdb_path = temp_dir / "DC" / "test.lmdb"
            for property_name in self.properties:
                if property_name != 'DC':
                    prop_dir = temp_dir / property_name
                    prop_dir.mkdir(exist_ok=True)
                    prop_lmdb_path = prop_dir / "test.lmdb"
                    if not prop_lmdb_path.exists():
                        os.symlink(dc_lmdb_path, prop_lmdb_path)
            
            # Run inference for each property
            predictions = {}
            for property_name in self.properties:
                weight_path = str(self.weight_dir / property_name / "ckpt" / property_name / "checkpoint_best.pt")
                
                logger.info(f"Predicting {property_name}...")
                psmi_list, pred_list = run_inference(
                    property_name, 
                    str(temp_dir), 
                    weight_path,
                    batch_size=self.batch_size,
                    use_gpu=self.use_gpu,
                    gpu_id=self.gpu_id
                )
                
                predictions[property_name] = pred_list
            
            # Combine predictions into list of dicts
            result = []
            for i in range(num_samples):
                pred_dict = {prop: float(predictions[prop][i]) for prop in self.properties}
                result.append(pred_dict)
            
            return result
        
        finally:
            # Cleanup temporary files
            if self.cleanup_cache and temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def save_cache(self, filepath: str):
        """Save prediction cache to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.prediction_cache, f)
        logger.info(f"Saved cache with {len(self.prediction_cache)} entries to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load prediction cache from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.prediction_cache = pickle.load(f)
            logger.info(f"Loaded cache with {len(self.prediction_cache)} entries from {filepath}")
        else:
            logger.warning(f"Cache file not found: {filepath}")
    
    def __del__(self):
        """Cleanup temporary cache directory if owned."""
        if hasattr(self, 'owns_cache_dir') and self.owns_cache_dir:
            if hasattr(self, 'cache_dir') and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)


def create_mmpolymer_predictor(
    weight_dir: str = '/internfs/Zy/dataset/finetune_data',
    properties: List[str] = ['Tg', 'DC'],
    **kwargs
) -> callable:
    """
    Factory function to create a PolyGA-compatible prediction function.
    
    This is the recommended way to create a predictor for use with PolyGA.
    
    Args:
        weight_dir: Directory containing MMPolymer weight files
        properties: List of properties to predict
        **kwargs: Additional arguments passed to MMPolymerPredictor
    
    Returns:
        Prediction function compatible with PolyGA's PolyPlanet
    
    Example:
        >>> predict_fn = create_mmpolymer_predictor(
        ...     weight_dir='/path/to/weights',
        ...     properties=['Tg', 'DC'],
        ...     batch_size=64
        ... )
        >>> planet = PolyPlanet(
        ...     name='test',
        ...     predict_function=predict_fn,
        ...     ...
        ... )
    """
    predictor = MMPolymerPredictor(
        weight_dir=weight_dir,
        properties=properties,
        **kwargs
    )
    return predictor.predict


# ============================================================================
# Standalone Prediction Function (for testing)
# ============================================================================

def predict_polymer_properties(
    psmiles_list: List[str],
    properties: List[str] = ['Tg', 'DC'],
    weight_dir: str = '/internfs/Zy/dataset/finetune_data',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Standalone function to predict properties for a list of polymer SMILES.
    
    Args:
        psmiles_list: List of polymer SMILES strings
        properties: Properties to predict
        weight_dir: Directory containing weight files
        output_path: Optional path to save results CSV
    
    Returns:
        DataFrame with predictions
    """
    predictor = MMPolymerPredictor(weight_dir=weight_dir, properties=properties)
    
    df = pd.DataFrame({'smiles_string': psmiles_list})
    result_df = predictor.predict(df, [], None)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return result_df


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MMPolymer Prediction for PolyGA")
    parser.add_argument("--input", required=True, help="PSMILES string or CSV file")
    parser.add_argument("--properties", default="Tg,DC", help="Comma-separated properties")
    parser.add_argument("--weight_dir", default="/internfs/Zy/dataset/finetune_data", 
                       help="Weight directory")
    parser.add_argument("--output", default="prediction_output.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--no_gpu", action='store_true', help="Disable GPU")
    args = parser.parse_args()
    
    # Parse properties
    properties = [p.strip() for p in args.properties.split(',')]
    
    # Load input
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        if 'smiles_string' not in df.columns and df.columns[0]:
            df = df.rename(columns={df.columns[0]: 'smiles_string'})
        psmiles_list = df['smiles_string'].tolist()
    else:
        psmiles_list = [args.input]
    
    print(f"Predicting {properties} for {len(psmiles_list)} polymers...")
    
    # Create predictor
    predictor = MMPolymerPredictor(
        weight_dir=args.weight_dir,
        properties=properties,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )
    
    # Predict
    df_input = pd.DataFrame({'smiles_string': psmiles_list})
    df_result = predictor.predict(df_input, [], None)
    
    # Save results
    df_result.to_csv(args.output, index=False)
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)
    print(df_result)
    print(f"\nResults saved to: {args.output}")

