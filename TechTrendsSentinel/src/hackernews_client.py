import os

import requests
from bs4 import BeautifulSoup
from logger import LOG
from datetime import datetime, date, timedelta  # 导入日期处理模块
from time import sleep

class HackerNewsClient:
    def __init__(self):
        pass
    def fetch_top_stories(self):
        url = "https://news.ycombinator.com/news"

        try:
            # 发送请求并获取页面内容
            response = requests.get(url)
            response.raise_for_status()  # 如果请求失败，抛出异常

            LOG.info("Successfully fetched top stories page.")

            # 解析页面
            soup = BeautifulSoup(response.text, 'html.parser')

            # 获取Top Stories
            stories = soup.find_all('tr', class_='athing submission')

            top_stories = [story.get_text() for story in stories]

            LOG.info(f"Fetched {len(top_stories)} top stories.")

            return top_stories

        except requests.exceptions.RequestException as e:
            LOG.error(f"Error fetching the page: {e}")
            return None
        except Exception as e:
            LOG.error(f"An unexpected error occurred: {e}")
            return None


    def save_to_markdown(self,top_stories):
        # 获取当前日期
        today_date = datetime.now().strftime('%Y-%m-%d-%H-%M')

        # 创建文件夹，如果不存在的话
        folder_path = 'hackernews_trends'
        os.makedirs(folder_path, exist_ok=True)

        # 创建文件路径，文件名为日期
        file_path = os.path.join(folder_path, f"top_stories_{today_date}.md")

        # 开始写入Markdown文件
        with open(file_path, 'w', encoding='utf-8') as file:
            # 写入文件头，包含日期
            file.write(f"# Hacker News Top Stories - {today_date}\n\n")

            # 写入每个标题
            for idx, title in enumerate(top_stories, 1):
                file.write(f"{title}\n\n")

        print(f"Markdown file saved at: {file_path}")
        return file_path


    def export_hackernews_top_stories(self):
        top_stories = self.fetch_top_stories()
        file_path=self.save_to_markdown(top_stories)
        return file_path



def main():
    hc=HackerNewsClient()
    top_stories_path = hc.export_hackernews_top_stories()
    if not top_stories_path:
        LOG.error("Failed to retrieve top stories.")


if __name__ == "__main__":
    main()
