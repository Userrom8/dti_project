# DTI-GNN-ESM

## Setup
1. Create conda env (recommended):
   conda create -n dti python=3.10
   conda activate dti

2. Install PyTorch + CUDA following https://pytorch.org

3. Install RDKit (conda recommended) or use `pip install rdkit-pypi` (may vary)
   conda install -c conda-forge rdkit

4. Install requirements:
   pip install -r requirements.txt

5. Prepare data:
   Put CSV files with columns: smiles, protein_sequence, affinity in the `data/` folder. For BindingDB, convert reported affinities to a single numeric scale (e.g., pKd = -log10(Kd)).

6. Train:
   python src/train.py --csv_path data/bindingdb.csv --epochs 10 --batch_size 8

7. Run API:
   uvicorn src.api:app --host 0.0.0.0 --port 8000

8. Run frontend:
   cd frontend
   npm install
   npm start
