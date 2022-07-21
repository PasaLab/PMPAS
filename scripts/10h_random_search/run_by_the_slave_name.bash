#!/bin/bash
fuck="210 216 287 505 507 574 42688 42730 42726 41021"

cd ../..
host_name=$(hostname)
for d in ${fuck}; do
  nohup python -m benchmarks.random_search_reg -s DAG -t 36000 -d $d >$d.txt 2>&1 &
done

logs_dir="reg_rs_dag_10h_logs"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
