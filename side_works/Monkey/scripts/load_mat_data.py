from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.io.matlab')

# 使用 _logs 目录下的正确 .mat 文件（这些是有效的 MATLAB v5 格式）
file_path = "/media/ubuntu/sda/Monkey/TVSD/monkeyF/_logs/THINGS_monkeyF_20240118_B1.mat"

print(f"读取文件: {file_path}")
a = loadmat(file_path)

print("\n数据加载完成!")
print(f"变量总数: {len(a)}")
print("\n主要变量预览:")
print(f"  - __header__: {str(a['__header__'])[:80]}...")
print(f"  - Par: {a['Par'].shape if hasattr(a['Par'], 'shape') else type(a['Par'])}")
print(f"  - LOG: {a['LOG'].shape if hasattr(a['LOG'], 'shape') else type(a['LOG'])}")
print(f"  - RANDTAB: {a['RANDTAB'].shape if hasattr(a['RANDTAB'], 'shape') else type(a['RANDTAB'])}")
print(f"  - TrialPic: {a['TrialPic'].shape if hasattr(a['TrialPic'], 'shape') else type(a['TrialPic'])}")
print(f"  - Hit: {a['Hit'].shape if hasattr(a['Hit'], 'shape') else type(a['Hit'])}")

# 显示所有变量名
print(f"\n完整变量列表:")
for i, key in enumerate(sorted(a.keys())):
    print(f"  {i+1}. {key}")
