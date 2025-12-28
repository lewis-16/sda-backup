# preprocess.py 输入输出格式说明

## 输入格式

### 主要输入：CSV文件

**CSV文件格式要求：**
- **格式**：基因表达矩阵（count matrix）
- **第一列**：基因名称（gene names/symbols）
  - 默认列名：`'Unnamed: 0'` 或需要重命名为 `'cell_id'`
- **其他列**：每个细胞（cell）的表达数据
  - 列名：细胞ID
  - 值：基因表达值（原始count值）

**CSV示例结构：**
```
Unnamed: 0,cell1,cell2,cell3,...
GAPDH,100,200,150,...
ACTB,50,75,60,...
...
```

**处理流程：**
1. 读取CSV：`pd.read_csv("./GSE210543_RPE_adult_counts.csv")`
2. 重命名第一列：`df.rename(columns={'Unnamed: 0': 'cell_id'})`
3. 设置索引并转置：`.set_index('cell_id').T`
   - 转置后：**行（index）= 基因名称，列（columns）= 细胞ID**
4. 转换为AnnData对象：`sc.AnnData(df)`
   - `adata.var.index` = 基因名称列表
   - `adata.obs.index` = 细胞ID列表
   - `adata.X` = 表达矩阵（细胞 × 基因）

### 其他必需输入：

1. **基因名称列表文件**：
   - `mouse_protein_coding.txt` / `human_protein_coding.txt` - 蛋白质编码基因列表
   - `mouse_miRNA.txt` / `human_miRNA.txt` - miRNA基因列表
   - `mouse_mitochondria.xlsx` / `human_mitochondria.xlsx` - 线粒体基因列表

2. **字典文件**：
   - `Gene_id_name_dict1.pickle` - 基因名称到ID的映射字典
   - `human_mouse_tokens.pickle` - 基因ID到token的映射字典
   - `human_gene_median_after_filter.pickle` - 基因中值字典（用于标准化）
   - `gene_id_hpromoter.pickle` - 基因ID列表（用于过滤）

## 输出格式

### 主要输出：HuggingFace Dataset格式

**保存位置：**
- 目录：`../scdata/output/patch{id}/`
- 例如：`../scdata/output/patch1/`

**输出内容：**

1. **Dataset目录结构：**
   ```
   patch1/
   ├── dataset_info.json
   ├── state.json
   └── *.arrow (parquet格式的数据文件)
   ```

2. **Dataset特征（Features）：**
   - `input_ids`: Sequence(Value(dtype='int32'))
     - 基因token ID序列，每个细胞最多2048个token
     - 按表达值降序排列的基因token
   - `values`: Sequence(Value(dtype='float32'))
     - 对应基因的表达值序列
     - 与input_ids一一对应，同样按降序排列
   - `length`: Sequence(Value(dtype='int16'))
     - 每个细胞实际使用的基因数量（非零基因数，最多2048）
   - `species`: Sequence(Value(dtype='int16'))
     - 物种标签：0=human, 1=mouse

3. **sorted_length.pickle：**
   - 所有细胞长度的排序列表
   - 用于后续的批次训练优化

**数据转换流程：**

1. **标准化**（Normalized）：
   - 使用基因中值字典进行标准化
   - 公式：`matrix / gene_median`

2. **Log转换**（log1p）：
   - 以2为底的对数转换
   - `sc.pp.log1p(adata, base=2)`

3. **Rank编码**（rank_value）：
   - 按表达值降序排列基因
   - 转换为token ID序列
   - 截断到2048个基因（如果超过）

4. **转换为Dataset：**
   - 使用HuggingFace的`Dataset.from_dict()`创建
   - 保存为磁盘格式供后续训练使用

## 与pretrain_test.py的区别

- **pretrain_test.py**：
  - 输入：已经预处理好的CSV（标准化、log转换已完成）
  - 直接读取并排序，不进行标准化
  - 输出：相同的Dataset格式

- **preprocess.py**：
  - 输入：原始count矩阵CSV
  - 进行完整预处理流程（过滤、标准化、log转换等）
  - 输出：相同的Dataset格式

## 使用示例

```python
# 输入CSV格式示例（原始count矩阵）
# 第一列是基因名称，其他列是细胞
df = pd.read_csv("input.csv")
# 列：Unnamed: 0, cell1, cell2, cell3, ...
# 行：GAPDH, ACTB, ...

# 输出：可以直接用于模型训练的Dataset
dataset = load_from_disk("../scdata/output/patch1/")
# dataset包含：input_ids, values, length, species
```

