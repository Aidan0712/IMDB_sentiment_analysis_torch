import os
import shutil
import hashlib
from sys import platform
from gensim.models import KeyedVectors

def prepend_slow(infile, outfile, line):
    """在文件开头插入一行内容"""
    with open(infile, 'r', encoding='utf-8') as fin:
        with open(outfile, 'w', encoding='utf-8') as fout:
            fout.write(line + "\n")
            shutil.copyfileobj(fin, fout)

def checksum(filename):
    """计算文件MD5校验"""
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

# 已知的GloVe文件行数和校验
pretrain_num_lines = {"glove.840B.300d.txt": 2196017}
pretrain_checksum = {"glove.840B.300d.txt": "eec7d467bccfa914726b51aac484d43a"}

def check_num_lines_in_glove(filename, check_checksum=False):
    if check_checksum:
        assert checksum(filename) == pretrain_checksum[filename]
    return pretrain_num_lines.get(filename, None)

# 输入文件（改成你自己的路径）
glove_file = r"F:\CodeSoft\glove.840B.300d\glove.840B.300d.txt"

# 输出文件
gensim_file = r"F:\CodeSoft\glove.840B.300d\glove.840B.300d.gensim.txt"

# 自动获取维度
num_lines = check_num_lines_in_glove(os.path.basename(glove_file))
dims = 300  # 840B版本默认是300维

# 写入第一行
gensim_first_line = f"{num_lines} {dims}"
prepend_slow(glove_file, gensim_file, gensim_first_line)

# 加载模型验证
print("正在加载 gensim 模型，请稍等（可能需要几分钟）...")
model = KeyedVectors.load_word2vec_format(gensim_file, binary=False)

# 简单测试
print(model.most_similar(positive=['australia'], topn=5))
print("相似度 woman-man：", model.similarity('woman', 'man'))
