#!/bin/bash
slave210="201 216 287 416"
slave211="422 505 507 531"
slave212="541 546 550 574"
slave214="3050 3277 41021 41540"
slave215="41702 41980 42225 42563"
slave216="42570 42688 42705 42724"
slave217="42726 42730 42731"

source activate bdas
cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.ea_reg_cv -s DAG -d $d >$d.txt 2>&1 &
done

logs_dir="ea_reg_100"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
