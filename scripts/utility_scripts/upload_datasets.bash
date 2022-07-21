#!/bin/bash
master02="210.28.132.12"

datasets_path="/home/qqq/PycharmProjects/datasets"

cd ../../..

tar -zcvf datasets.tar.gz ${datasets_path}

echo "Begin to upload files..."
sshpass -p letmetry scp datasets.tar.gz experiment@${master02}:~/qijiahao
echo "Upload finished..."