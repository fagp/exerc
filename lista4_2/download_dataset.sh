#cd /content/
pip install gdown
gdown https://drive.google.com/uc?id=1qQQ6sryoooQP8RLMdffV7vloT4U1mD_U
unzip cells_dataset.zip
mv cells_dataset train
mkdir val
mkdir val/images/
mkdir val/labels/

cd train/images/
for file in $(ls -p | grep -v / | tail -100)
do
mv $file ../../val/images/
done

cd ../labels/
for file in $(ls -p | grep -v / | tail -100)
do
mv $file ../../val/labels/
done

rm ../../cells_dataset.zip
