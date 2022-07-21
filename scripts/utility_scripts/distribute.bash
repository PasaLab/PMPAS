cd ../../..

slaves="210 211 212 213 214 215 216 217 219 220 221 222 223 224 225 226 227 228 229"


for slave in $slaves;do
  echo "transfer to slave$slave";
  scp qi.tar.gz slave${slave}:~/qijiahao
done

