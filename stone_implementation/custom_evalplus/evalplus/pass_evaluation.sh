#! /bin/bash 

source ~/.bashrc 

conda activate stone

date 

############unwatermarked solutions
for hash_key in 15485863 
do
    for method in "STONE" ##"STONE" "SWEET" "EWD" "KGW"
    do
        for model in "qwen" #"llama" "qwen"
        do
            for language in "python" #"cpp" "java" 
            do
                for data in "mbpp" #"humaneval" #"mbpp" 
                do 
                    python sanitize.py --samples ../../results/1samples/${data}plus_${language}_${model}_${method}_${hash_key}__unwatermarked_solutions.jsonl 
                    python evaluate.py --dataset ${data} --samples ../../results/1samples/${data}plus_${language}_${model}_${method}_${hash_key}__unwatermarked_solutions-sanitized.jsonl 
                done
            done
        done
    done
done



##############humaneval
for delta in 0.5  
do 
    for gamma in 0.5
    do 
        for hash_key in 15485863 
        do 
            for watermark_on_pl in "False"  
            do 
                for method in "EWD" #"EWD" "SWEET" "EWD" "KGW"
                do
                    for model in "qwen" #"llama" #"qwen"
                    do 
                        for skipping_rule in "all_pl"
                        do
                            for language in "mbpp" #"cpp" "java" 
                            do                       
                                for data in "humaneval"
                                do
                                    python sanitize.py --samples ../../results/1samples/${data}plus_${language}_${model}_${method}_${skipping_rule}_${watermark_on_pl}_${gamma}_${delta}_${hash_key}__watermarked_solutions.jsonl
                                    python evaluate.py --dataset $data --samples ../../results/1samples/${data}plus_${language}_${model}_${method}_${skipping_rule}_${watermark_on_pl}_${gamma}_${delta}_${hash_key}__watermarked_solutions-sanitized.jsonl
                                done
                            done
                        done
                    done
                done 
            done 
        done 
    done 
done 


#############mbpp
for delta in 1.0
do 
    for gamma in 0.5
    do 
        for hash_key in 15485863 
        do 
            for watermark_on_pl in "False"  
            do 
                for method in "EWD" #"KGW" #"SWEET" "EWD" "KGW"
                do
                    for model in "qwen" #"llama" "qwen"
                    do 
                        for skipping_rule in "all_pl"
                        do
                            for language in "python" #"cpp" "java" 
                            do                       
                                for data in "mbpp"
                                do
                                    python sanitize.py --samples ../../results/final/1samples/${data}plus_${language}_${model}_${method}_${skipping_rule}_${watermark_on_pl}_${gamma}_${delta}_${hash_key}__watermarked_solutions.jsonl
                                    python evaluate.py --dataset $data --samples ../../results/1samples/${data}plus_${language}_${model}_${method}_${skipping_rule}_${watermark_on_pl}_${gamma}_${delta}_${hash_key}__watermarked_solutions-sanitized.jsonl
                                done
                            done
                        done
                    done
                done 
            done 
        done 
    done 
done 





date 

rm -rf __pycache__