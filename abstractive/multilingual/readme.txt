
git clone https://github.com/user/zen.git
cd zen/
git pull origin main

cd mbart_baseline/
conda create --name mbart_baseline python==3.8
conda activate mbart_baseline
pip install -r requirements.txt
bash run_local.sh


