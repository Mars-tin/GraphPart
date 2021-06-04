module load python/3.7.4
module load cudnn/10.1-v7.6
module load cuda/10.1.105

python3 -m venv env
source env/bin/activate	
	
pip install pip==20.2.4
pip install numpy
pip install pandas
pip install dgl
pip install scikit-learn
pip install scikit-learn-extra

pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric \
  torch-sparse==latest+cu101 \
  torch-scatter==latest+cu101 \
  torch-cluster==latest+cu101 \
  -f https://pytorch-geometric.com/whl/torch-1.7.0.html
