cd ../../..

slaves="212 214 215 216 217 219 220 221 222 223 224 225 226 227"


for slave in $slaves;do
  echo "transfer to slave$slave";
  scp qi.tar.gz slave${slave}:~/qijiahao
done

for slave in $slaves;do
  echo "begin to extract files...";
  ssh slave$slave "cd ~/qijiahao;tar -zxvf qi.tar.gz"
done