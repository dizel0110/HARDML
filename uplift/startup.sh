#!/bin/bash

mkdir -p ~/hard-ml/uplift
git clone https://github.com/Antiguru11/hard-ml.git ~/hard-ml/uplift

git config --global user.name "Antiguru11"
git config --global user.email "antiguru110894@gmail.com"

cd ~/hard-ml/uplift/final 
export LANG=C.UTF-8
DATA_URL=$(curl -X GET --header 'Accept: application/json' --header 'Authorization: OAuth $1' 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https%3A%2F%2Fdisk.yandex.ru%2Fd%2F2LaYWIY5jb_Mcw' | awk -F \" '{print $4}')
wget "$DATA_URL" -O data.zip
unzip data.zip -d data
mv 'data/Финальный проект'/* data
rm -fR 'data/Финальный проект'
rm -f data.zip