slave220="3 6 11 12 14 16 18"
slave221="22 23 28 29 31 32 37"
slave222="38 46 50 54 151 182 188"
slave224="300 307 458 1049 1050 1053 1063"
slave225="1462 1464 1468 1478 1489 1494 1497"
slave226="1501 1510 1590 4534 4538 23381 40499"
slave227="40668 40670 40701 40966 40975 40979 40982"
slave228="40983 40984 40994 41027"

cd ../..
host_name=$(hostname)
for d in ${!host_name}; do
  nohup python -m benchmarks.ea_iter_cv -s plain  -d $d >${d}.txt 2>&1 &
done

logs_dir="ea_plain_100"
if [ ! -d "$logs_dir" ]; then
  mkdir $logs_dir
fi

mv [[:digit:]]*.txt $logs_dir
