import re

src_html = '<p>just for test</p><br/><font>just for test</font><b>test</b>'
# pat = re.compile('(?<=\>).*?(?=\<)')

# if __name__ == '__main__':
#     print(pat.findall(src_html))

def clear_html_re(src_html):
    '''
    正则清除HTML标签
    :param src_html:原文本
    :return: 清除后的文本
    '''
    content = re.sub(r"</?(.+?)>", "", src_html) # 去除标签, '.'表示 除了换行符之外的任意字符
    # content = re.sub(r"&nbsp;", "", content)
    dst_html = re.sub(r"\s+", "", content)  # 去除空白字符
    return dst_html

if __name__ == '__main__':
    print(clear_html_re(src_html))
