# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface

from itemadapter import ItemAdapter
import psycopg2
from scrapy.exceptions import DropItem


class BooksScraperPipeline:

    def open_spider(self, spider):
        self.connection = psycopg2.connect(
            host='localhost',
            user='postgres',
            password='postgres',
            database='books_database'
        )
        self.cursor = self.connection.cursor()

    def close_spider(self, spider):
        self.cursor.close()
        self.connection.close()

    def process_item(self, item, spider):
        try:
            self.cursor.execute("""
                        INSERT INTO books (code, title, author, price, category, publisher, binding, year, format, pages, description) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                item.get('code'),
                item.get('title'),
                item.get('author'),
                item.get('price'),
                item.get('category'),
                item.get('publisher'),
                item.get('binding'),
                item.get('year'),
                item.get('format'),
                item.get('pages'),
                item.get('description')
            ))
            self.connection.commit()
        except psycopg2.Error as e:
            spider.logger.error(f"Error: {e}")
            raise DropItem(f"Error inserting item: {e}")

        return item
