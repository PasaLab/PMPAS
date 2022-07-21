#!/bin/bash
slave221="210 216 287 505 507 574 42688 42730 42726 41021"

cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.ea_time_reg -s DAG -t 36000 -d $d >$d.txt 2>&1 &
done

logs_dir="reg_ea_dag_10h_logs"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
