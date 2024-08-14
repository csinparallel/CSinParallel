
#!/bin/bash


shopt -s nullglob   # empty directory will return empty list
for dir in ./*/;do
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
    cd ..
    echo "============================================================="
done
