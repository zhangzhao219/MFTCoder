#pip install transformers==4.32.0

N_NODE=1

N_GPU_PER_NODE=1


tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)


model=/home/aiscuser/Code-Qwen/output/20231209/merge/step_38000
model_name=20231209-step_38000
generation_base_dir=/home/aiscuser/Code-Qwen/output/20231209/generate/step_38000


if [ ! -d $generation_base_dir ]; then
    mkdir $generation_base_dir
fi


batch_size=1
n_samples=1
# For qwen base model, eos is '<|endoftext|>'; for fine-tuned qwen model, eos is '<|im_end|>'
eos_token="<|im_end|>"


# SFT Format
user=user
assistant=assistant
system=system
end_tag="<|im_end|>"
start_tag="<|im_start|>"

# If you need to set system prompt, set it here, otherwise you can set it as empty string. Decide whether to add system prompt by yourself.
if [ "$task" == "mbpp" ]; then
    system="$start_tag"$system$'\n'"You are an expert code programmer, now you should answer code question below, please give your code directly"$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-python" ]; then
    system="$start_tag"$system$'\n'"You are an expert Python programmer, Now I will give you a code question and part of the function in Python, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-java" ]; then
    system="$start_tag"$system$'\n'"You are an expert Java programmer, Now I will give you a code question and part of the function in Java, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-js" ]; then
    system="$start_tag"$system$'\n'"You are an expert JavaScript programmer, Now I will give you a code question and part of the function in JavaScript, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-cpp" ]; then
    system="$start_tag"$system$'\n'"You are an expert C++ programmer, Now I will give you a code question and part of the function in C++, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-go" ]; then
    system="$start_tag"$system$'\n'"You are an expert Go programmer, Now I will give you a code question and part of the function in Go, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalsynthesize-rust" ]; then
    system="$start_tag"$system$'\n'"You are an expert Rust programmer, Now I will give you a code question and part of the function in Rust, you should provide me with the complete function that work normally."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-python" ]; then
    system="$start_tag"$system$'\n'"You are an expert Python programmer, Now I will give you a piece of a code in Python, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-java" ]; then
    system="$start_tag"$system$'\n'"You are an expert Java programmer, Now I will give you a piece of a code in Java, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-js" ]; then
    system="$start_tag"$system$'\n'"You are an expert JavaScript programmer, Now I will give you a piece of a code in JavaScript, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-cpp" ]; then
    system="$start_tag"$system$'\n'"You are an expert C++ programmer, Now I will give you a piece of a code in C++, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-go" ]; then
    system="$start_tag"$system$'\n'"You are an expert Go programmer, Now I will give you a piece of a code in Go, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
elif [ "$task" == "humanevalfixtests-rust" ]; then
    system="$start_tag"$system$'\n'"You are an expert Rust programmer, Now I will give you a piece of a code in Rust, but there is a bug in it, you should fix the bug with test cases provided by me."$'\n'"$end_tag"$'\n'
fi

for task in "${tasks[@]}"; do

    prefix="$system""$start_tag"${user}$'\n'
    suffix="$end_tag"$'\n'"$start_tag"${assistant}'\n'

    generations_path=$generation_base_dir/generations_$model_name/generations_$task\_$model_name.json
    if [ ! -d $generation_base_dir/generations_$model_name ]; then
        mkdir $generation_base_dir/generations_$model_name
    fi

    echo "start to launch ...."
    accelerate launch \
            --num_machines $N_NODE \
            --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
            main.py \
                --model $model \
                --task $task \
                --prompt instruct \
                --n_samples $n_samples \
                --batch_size $batch_size \
                --max_length_generation 2000 \
                --do_sample False \
                --temperature 0.2 \
                --precision bf16 \
                --eos "$eos_token" \
                --seed 999999999 \
                --add_special_tokens True \
                --trust_remote_code \
                --generation_only \
                --save_generations_path $generations_path \
                --prefix "$prefix" \
                --suffix "$suffix"
    
    echo "Task $task done"
done

echo $model_name
