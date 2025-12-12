import datetime
import time
import requests
from lxml import etree

cookies = {
    "Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2": "1653439907",
    "BAIDU_SSP_lcr": "https://www.baidu.com/link?url=CjJZmL3qBJKe2EX3TpNd-1RtPBsScgPBbt41kht0HRwaJ39P-Atlk4Y4gUPtsNDx&wd=&eqid=c6f3673400002f4500000005628d88bc",
    "Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2": "1653447163",
}

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    # Requests sorts cookies= alphabetically
    # 'Cookie': 'Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1653439907; BAIDU_SSP_lcr=https://www.baidu.com/link?url=CjJZmL3qBJKe2EX3TpNd-1RtPBsScgPBbt41kht0HRwaJ39P-Atlk4Y4gUPtsNDx&wd=&eqid=c6f3673400002f4500000005628d88bc; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1653447163',
    "DNT": "1",
    # 'If-Modified-Since': 'Tue, 24 May 2022 13:40:22 GMT',
    "Referer": "https://lishi.tianqi.com/shanghai/201501.html",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

total_result = []

for year in range(2011, 2023):
    for month in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]:
        if year == 2022 and month in [
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]:
            continue
        time.sleep(2)
        url = f"https://lishi.tianqi.com/shanghai/{year}{month}.html"
        rp = requests.get(url, cookies=cookies, headers=headers)
        if rp.status_code != 200:
            raise Exception()
        html = etree.HTML(rp.text)
        data_list = html.xpath('//ul[@class="thrui"]/li/div/text()')
        print(year, month, int(len(data_list) / 5))
        for dt, temperature_max, temperature_min, weather in zip(
            data_list[0::5], data_list[1::5], data_list[2::5], data_list[3::5]
        ):
            date = dt.strip().split(" ")[0]
            week = datetime.date.fromisoformat(date).weekday()

            temperature_max = float(temperature_max[:-1])
            temperature_min = float(temperature_min[:-1])
            weather = weather.strip()
            total_result.append(
                {
                    "date": date,
                    "week": week,
                    "temperature_max": temperature_max,
                    "temperature_min": temperature_min,
                    "weather": weather,
                }
            )
