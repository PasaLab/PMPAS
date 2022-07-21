#!/bin/bash
cd ../../..
up_files="qi"
file_names="qi.tar.gz"

master02="210.28.132.12"

# todo ,add exclude files
echo "Begin to tar files..."
tar -zcvf $file_names ${up_files} --exclude=experiment_results --exclude=.git --exclude=ex.tar.gz --exclude=results --exclude=fuck.bash --exclude=fuck.py --exclude=core/datasets
echo "Tar files finished..."

echo "Begin to upload files..."
sshpass -p letmetry scp ${file_names} experiment@${master02}:~/qijiahao
echo "Upload finished..."
