import sys
from pathlib import Path


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # 保存原始的标准输出流
        self.logfile = open(filename, 'w')  # 打开文件以写入输出
        sys.stdout = self  # 重定向标准输出流到该对象

    def write(self, message):
        self.terminal.write(message)  # 屏幕输出
        self.logfile.write(message)  # 文件输出

    def close(self):
        sys.stdout = self.terminal  # 还原标准输出流
        self.logfile.close()  # 关闭文件

    def flush(self):
        pass  # 添加 flush 方法以解决警告

def set_logger(filename):
    filename = Path(filename)
    if not Path.exists(filename.parent):
        Path(filename.parent).mkdir()
    filename.touch()
    return Logger(filename)

if __name__=="__main__":
    # # 使用自定义 Logger 类
    # logger = Logger('output.txt')
    # # 打印输出，同时写入屏幕和文件
    # print("This is written to the screen and file.")
    # # 关闭 Logger
    # logger.close()
    # # 现在 print 的输出会回到屏幕
    # print("This is written only to the screen.")

    set_logger('./logs/a/1.txt')
    print(123)
    print(123)
    print(123)
    print(123)
