import sys
def add(*numbers):
    """
    实现一个多个数字的相加处理操作
    :param numbers: 要进行加法操作的数字内容
    :return: 累加的结果
    """
    sum = 0
    for num in numbers:
            sum += num
    return sum

