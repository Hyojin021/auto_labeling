#!/bin/sh

brew install python@2
pip install --upgrade virtualenv

# clone udk-label source
rm -rf /tmp/udk-label-setup
mkdir /tmp/udk-label-setup
cd /tmp/udk-label-setup
curl https://codeload.github.com/tzutalin/udk-label/zip/master --output udk-label.zip
unzip udk-label.zip
rm udk-label.zip

# setup python3 space
virtualenv --system-site-packages  -p python3 /tmp/udk-label-setup/udk-label-py3
source /tmp/udk-label-setup/udk-label-py3/bin/activate
cd udk-label-master

# build udk-label app
pip install py2app
pip install PyQt5 lxml
make qt5py3
rm -rf build dist
python setup.py py2app -A
mv "/tmp/udk-label-setup/udk-label-master/dist/UDK Label.app" /Applications
# deactivate python3
deactivate
cd ../
rm -rf /tmp/udk-label-setup
echo 'DONE'
