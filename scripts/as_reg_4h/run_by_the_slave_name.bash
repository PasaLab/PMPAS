#!/bin/bash
slave217="201 216 287 505 507 574 42688 42730 42726 41021"

cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.as_reg_4h  -d $d >$d.txt 2>&1 &
done

logs_dir="as_reg_4h"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
