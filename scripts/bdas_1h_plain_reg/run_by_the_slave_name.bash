#!/bin/bash
slave221="201 216 287 416 422 505 507"
slave222="531 541 546 550 574 3050 3277"
slave224="41021 41540 41702 41980 42225"
slave225="42563 42570 42688 42705 42724 42726"
slave227="42730 42731"

source activate bdas
cd ../..
host_name=slave227
for d in ${!host_name}; do
  nohup python -m benchmarks.bdas_reg_time_cv -s plain  -d $d >$d.txt 2>&1 &
done

logs_dir="bdas_reg_plain_1h"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
