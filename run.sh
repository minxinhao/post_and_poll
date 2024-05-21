clang++ -std=c++20 -libverbs -lpthread  verbtest.cc -O3
rm -f out.txt
for num_cli in 12 24 36 48 60 72 84 96;do
    for depth in 1 4 8 16 24 32;do
        ./a.out $num_cli $depth >out.txt
    done
done