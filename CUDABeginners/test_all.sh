
#!/bin/bash

module load nvhpc/24.7

shopt -s nullglob   # empty directory will return empty list
for dir in ./*/;do
    echo "$dir"         # dir is directory only because of the / after *
    if [ "$dir" = "./1-basics/" ];
    then 
        continue
    fi
    cd $dir
    make clean
    make
    if test -f run.test; then
        echo "test file exists."
        shelltest -c run.test
    else
        echo "no test file yet; skipping."
    fi
   
    make clean
    cd ..
    echo "============================================================="
done

for dir in ./*/*/;do
    echo "$dir"         # dir is directory only because of the / after *
    cd $dir
    make clean
    make
    if test -f run.test; then
        echo "test file exists."
        shelltest -c run.test
    else
        echo "no test file yet; skipping."
    fi
   
    make clean
    cd ../..
    echo "============================================================="
done

module unload nvhpc/24.7
