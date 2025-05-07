#! /bin/bash 

source ~/.bashrc 

conda activate stone

date 

################[humanevalpack]##################
for delta in 0.5 
do 
    for gamma in 0.5  
    do 
        for hash_key in 15485863
        do 
            for method in "STONE" "SWEET" "EWD" "KGW"  
            do 
                for model in "qwensmall" #"qwen"
                do 
                    for data in "humanevalpack"
                    do
                       for language in "cpp" "java" 
                        do
                            for n_samples in 5 
                            do
                                echo "Running with settings: delta=$delta, gamma=$gamma, hash_key=$hash_key, method=$method, model=$model, data=$data, language=$language, n_samples=$n_samples"
                                CUDA_VISIBLE_DEVICES=1 python run.py \
                                --model $model --data $data --method $method \
                                --hash_key $hash_key --gamma $gamma --delta $delta \
                                --language $language --n_samples $n_samples
                            done
                        done
                    done
                done 
            done 
        done 
    done 
done 




# ################[mbppplus]##################
for delta in 1.0  
do
    for gamma in 0.5 
    do 
        for hash_key in 15485863 
        do 
            for method in "STONE" "SWEET" "EWD" "KGW"  
            do 
                for model in "qwensmall" #"qwen"
                do 
                    for data in "mbppplus" 
                    do
                        for n_samples in 5
                        do
                            CUDA_VISIBLE_DEVICES=1 python run.py \
                            --delta $delta --gamma $gamma --hash_key $hash_key \
                            --method $method --model $model --data $data \
                            --n_samples $n_samples
                        done
                    done
                done
            done 
        done 
    done 
done 

# ################[humanevalplus]##################
for delta in 0.5   
do 
    for gamma in 0.5 
    do 
        for hash_key in 15485863 
        do 
            for method in "STONE" "SWEET" "EWD" "KGW"  
            do 
                for model in "qwensmall" #"qwen"
                do 
                    for data in "humanevalplus" 
                    do
                        for n_samples in 5
                        do
                            CUDA_VISIBLE_DEVICES=1 python run.py \
                            --delta $delta --gamma $gamma --hash_key $hash_key \
                            --method $method --model $model --data $data \
                            --n_samples $n_samples
                        done
                    done
                done
            done 
        done 
    done 
done 






date 

rm -rf __pycache__