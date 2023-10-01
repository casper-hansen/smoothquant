sh install_cutlass.sh
git clone https://github.com/casper-hansen/torch-int.git
cd torch-int
pip install -v -e .
cd ..
pip install -v -e .
pip install transformers accelerate datasets zstandard