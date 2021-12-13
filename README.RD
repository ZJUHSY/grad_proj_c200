####How to run?


###First install Fasttext
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
sudo pip install .
sudo python setup.py install

### run the Textcnn model for classification
cd ..
python run.py

Environment:
python == 3.6
torch == 1.10.0
fasttext == 0.9.2