#!/bin/bash
slave219="201 216 287 416 422 505 507"
slave221="531 541 546 550 574 3050 3277"
slave224="41021 41540 41702 41980 42225"
slave225="42563 42570 42688 42705 42724 42726"
slave227="42730 42731"

source activate bdas
cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.autosklearn_reg_1h_cv  -d $d >$d.txt 2>&1 &
done

logs_dir="as_reg_1h"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
