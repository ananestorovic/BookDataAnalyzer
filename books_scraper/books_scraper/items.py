# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BooksScraperItem(scrapy.Item):
    code = scrapy.Field()  # Šifra proizvoda
    title = scrapy.Field()  # Naslov knjige
    author = scrapy.Field()  # Autor knjige
    price = scrapy.Field()  # Cena knjige
    category = scrapy.Field()  # Kategorija knjige
    publisher = scrapy.Field()  # Izdavač knjige
    binding = scrapy.Field()  # Povez knjige
    year = scrapy.Field()  # Godina izdavanja
    format = scrapy.Field()  # Format knjige
    pages = scrapy.Field()  # Broj stranica
    description = scrapy.Field()  # Opis knjige
