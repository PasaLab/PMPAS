#!/bin/bash
slave211="201 216 287 416 422 505 507"
slave212="531 541 546 550 574 3050 3277"
slave214="41021 41540 41702 41980 42225"
slave215="42563 42570 42688 42705 42724 42726"
slave217="42730 42731"

source activate bdas
cd ../..
host_name=slave217
for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_reg_time_cv -s DAG  -d $d >$d.txt 2>&1 &
done

logs_dir="bdas_reg_1h"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
