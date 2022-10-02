# -*- coding: utf-8 -*
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则，文字匹配
import requests  # 获取网页数据
import xlwt  # excel操作


def main():
    baseUrl = "https://movie.douban.com/top250?start="
    # 爬取网页
    dataList = getDate(baseUrl)
    # 保存数据
    savePath = "豆瓣top250.xls"
    saveData(savePath, dataList)
    # saveDataToDb(dataList)


# 得到指定URL的网页内容
def askUrl(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'
    }
    html = ""
    try:
        r = requests.get(url=url, headers=headers, timeout=3)
        r.encoding = 'utf-8'
        html = r.text
    except Exception as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html


def getDate(baseUrl):
    dataList = []
    for i in range(0, 10):
        url = baseUrl + str(i * 25)
        html = askUrl(url)
        # 解析数据
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all("div", class_="item"):
            data = []  # 存放一部电影的所有信息
            item = str(item)
            link = re.findall(r'<a href="(.*)">', item)[0]  # 链接
            data.append(link)
            image = re.findall(r'<img.*src="(.*)" .*/>', item)[0]  # 图片
            data.append(image)
            titles = re.findall(r'<span class="title">(.*)</span>', item)  # 片名
            data.append(titles[0])  # 添加中文名
            if len(titles) == 2:  # 添加外国名
                data.append(titles[1].replace("\\", ""))
            else:
                data.append(" ")
            rate = re.findall(r'<span class="rating_num".*>(.*)</span>', item)[0]  # 评分
            data.append(rate)
            judge = re.findall(r'<span>(\d*)人评价</span>', item)[0]  # 评级人数
            data.append(judge)
            inq = re.findall(r'<span class="inq">(.*)</span>', item, re.S)  # 简述
            if len(inq) != 0:
                inq = inq[0].replace("。", "")
                data.append(inq)
            else:
                data.append("")
            bd = re.findall(r'<p class="">(.*?)</p>', item, re.S)[0]  # 其他信息
            bd = re.sub('<br/>', " ", bd)
            bd = re.sub("/", " ", bd)
            bd = re.sub("\\n", " ", bd)
            bd = re.sub(r"\xa0", " ", bd)
            data.append(bd.strip())
            dataList.append(data)
    return dataList


def saveData(savePath, dataList):
    workbook = xlwt.Workbook(encoding="utf-8", style_compression=0)
    worksheet = workbook.add_sheet("豆瓣top250", cell_overwrite_ok=True)
    col = ("电影详情链接", "图片链接", "影片中文名", "影片英文名", "评分", "评价数", "概况", "相关信息")
    for i in range(0, 8):
        worksheet.write(0, i, col[i])
    for i in range(0, 250):
        data = dataList[i]
        for j in range(0, 8):
            worksheet.write(i + 1, j, data[j])
    workbook.save(savePath)


if (__name__ == "__main__"):
    main()
