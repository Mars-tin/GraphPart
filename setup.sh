# Module Loading
module load python/3.7.4
module load cudnn/11.2-v8.1.0
module load cuda/11.2.1

# Python Environment
python3 -m venv env
source env/bin/activate	

# Dependencies
pip install pip==20.2.4
pip install matplotlib
pip install numpy
pip install pandas
pip install dgl
pip install scikit-learn
pip install scikit-learn-extra
pip install kneebow

# PyTorch
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

# Open Graph Benchmarks
pip install ogb

