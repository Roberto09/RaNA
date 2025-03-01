pip install pandas
pip install transformers==4.40.2
pip install datasets==2.16.1
pip install trl==0.8.6 
pip install peft==0.7.1  
pip install matplotlib==3.8.2

git clone https://github.com/EleutherAI/lm-evaluation-harness 
cd lm-evaluation-harness
git reset --hard 4d7d2f64576205105318fd12a622b6f0b7c70464
pip install -e .
