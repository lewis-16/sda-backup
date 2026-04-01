#!/bin/bash

output_base="/media/ubuntu/sda/all_output"
probe_position="/media/ubuntu/sda/data/probe.json"
python_script="/home/ubuntu/Documents/jct/project/code/kilosort_pepline.py"

mice=("mouse2" "mouse5" "mouse6" "mouse11")

stim_types=("grating" "natural_image")

for mouse in "${mice[@]}"; do
    if [ ! -d "./${mouse}" ]; then
        echo "警告：目录 ./${mouse} 不存在，跳过。"
        continue
    fi

    for stim_type in "${stim_types[@]}"; do
        input_dir="./${mouse}/ns4/${stim_type}"
        
        if [ ! -d "${input_dir}" ]; then
            echo "警告：输入目录 ${input_dir} 不存在，跳过。"
            continue
        fi

        # 遍历.ns4文件
        for file in "${input_dir}"/*.ns4; do
            [ -e "${file}" ] || continue

            filename=$(basename "${file}")
            if [[ ! "${filename}" =~ ^${mouse}_[0-9]+_.+\.ns4$ ]]; then
                echo "错误：文件 ${filename} 格式不符合要求，跳过。"
                continue
            fi

            date_part=$(echo "${filename}" | cut -d'_' -f2)
            
            output_path="${output_base}/${mouse}/${stim_type}/${date_part}/"
            
            mkdir -p "${output_path}"
            
            echo "正在处理：${file} -> ${output_path}"
            /home/ubuntu/.conda/envs/spike_sorting_jct/bin/python "${python_script}" "${file}" "${probe_position}" "${output_path}"
        done
    done
done

echo "全部处理完成！"