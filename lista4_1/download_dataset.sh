cd /content/
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017 train
mkdir val
cd train/
for file in $(ls -p | grep -v / | tail -1000)
do
mv $file ../val/
done
rm ../val2017.zip
