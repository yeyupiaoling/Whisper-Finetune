import json
import mmap

import struct

from tqdm import tqdm


class DatasetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.data_sum = 0
        self.offset = 0
        self.header = ''

    def add_data(self, data):
        key = str(self.data_sum)
        data = bytes(data, encoding="utf8")
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(data)))
        self.data_file.write(data)
        # 写入索引
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(data)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(data)
        self.data_sum += 1

    def close(self):
        self.data_file.close()
        self.header_file.close()


class DatasetReader(object):
    def __init__(self, data_header_path, min_duration=0, max_duration=30):
        self.keys = []
        self.offset_dict = {}
        self.fp = open(data_header_path.replace('.header', '.data'), 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        for line in tqdm(open(data_header_path, 'rb'), desc='读取数据列表'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            data = self.m[int(val_pos):int(val_pos) + int(val_len)]
            data = str(data, encoding="utf-8")
            data = json.loads(data)
            # 跳过超出长度限制的音频
            if data["duration"] < min_duration:
                continue
            if max_duration != -1 and data["duration"] > max_duration:
                continue
            self.keys.append(key)
            self.offset_dict[key] = (int(val_pos), int(val_len))

    # 获取一行列表数据
    def get_data(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        data = self.m[val_pos:val_pos + val_len]
        data = str(data, encoding="utf-8")
        return json.loads(data)

    # 获取keys
    def get_keys(self):
        return self.keys

    def __len__(self):
        return len(self.keys)
