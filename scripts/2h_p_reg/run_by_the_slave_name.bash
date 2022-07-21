#!/bin/bash
slave212="201 216 287 505 507 574 42688 42730 42726 41021"

cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_reg_time_cv -t 7200 -s DAG  -d $d >$d.txt 2>&1 &
done

logs_dir="bdas_reg_2h"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
