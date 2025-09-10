#!/bin/bash

output_base="/media/ubuntu/sda/data/sort_output"
probe_position="/media/ubuntu/sda/data/probe.json"
python_script="/home/ubuntu/Documents/jct/project/code/kilosort_pepline.py"

mice=("mouse15" "mouse21" "mouse23" 'mouse24')


for mouse in "${mice[@]}"; do
    echo "${mouse}"
    if [ ! -d "./${mouse}" ]; then
        echo "警告：目录 ./${mouse} 不存在，跳过。"
        continue
    fi

    
    input_dir="./${mouse}/ns4/"
    echo "${input_dir}"
    for file in "${input_dir}"/*.ns4; do
        [ -e "${file}" ] || continue

        filename=$(basename "${file}")
        if [[ ! "${filename}" =~ ^${mouse}_[0-9]+_.+\.ns4$ ]]; then
            echo "错误：文件 ${filename} 格式不符合要求，跳过。"
            continue
        fi

        date_part=$(echo "${filename}" | cut -d'_' -f2)
        
        output_path="${output_base}/${mouse}/${date_part}/"
        
        mkdir -p "${output_path}"
        
        echo "正在处理：${file} -> ${output_path}"
        /home/ubuntu/.conda/envs/spike_sorting_jct/bin/python "${python_script}" "${file}" "${probe_position}" "${output_path}"
    done
done

echo "全部处理完成！"